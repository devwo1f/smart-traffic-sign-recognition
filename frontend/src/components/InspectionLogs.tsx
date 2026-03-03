import { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../config';
import type { HistoryResponse, HistoryItem } from '../types';
import './InspectionLogs.css';

export default function InspectionLogs() {
    const [items, setItems] = useState<HistoryItem[]>([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Filters
    const [label, setLabel] = useState('');
    const [minConf, setMinConf] = useState('');
    const [maxConf, setMaxConf] = useState('');
    const [source, setSource] = useState('');
    const [dateFrom, setDateFrom] = useState('');
    const [dateTo, setDateTo] = useState('');

    const fetchHistory = useCallback(async (pageNum: number) => {
        setLoading(true);
        setError(null);

        const params = new URLSearchParams();
        params.set('page', String(pageNum));
        params.set('page_size', '50');

        if (label) params.set('label', label);
        if (minConf) params.set('min_confidence', minConf);
        if (maxConf) params.set('max_confidence', maxConf);
        if (source) params.set('source_type', source);
        if (dateFrom) params.set('date_from', new Date(dateFrom).toISOString());
        if (dateTo) params.set('date_to', new Date(dateTo).toISOString());

        try {
            const res = await fetch(`${API_BASE}/history?${params.toString()}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data: HistoryResponse = await res.json();
            setItems(data.items);
            setTotalPages(data.total_pages);
            setTotal(data.total);
            setPage(data.page);
        } catch (err: any) {
            setError(err.message || 'Failed to load history');
        } finally {
            setLoading(false);
        }
    }, [label, minConf, maxConf, source, dateFrom, dateTo]);

    // Load on mount
    useEffect(() => {
        fetchHistory(1);
    }, []);  // eslint-disable-line react-hooks/exhaustive-deps

    const handleFilter = () => { setPage(1); fetchHistory(1); };
    const clearFilters = () => {
        setLabel(''); setMinConf(''); setMaxConf('');
        setSource(''); setDateFrom(''); setDateTo('');
        setTimeout(() => fetchHistory(1), 0);
    };

    const confBadge = (conf: number) => {
        const cls = conf >= 0.8 ? 'high' : conf >= 0.5 ? 'medium' : 'low';
        return <span className={`badge badge-${cls}`}>{(conf * 100).toFixed(1)}%</span>;
    };

    const truncate = (s: string, n: number) => s.length > n ? s.slice(0, n) + '…' : s;

    return (
        <div className="logs-wrapper" style={{ animation: 'fadeIn 0.3s ease' }}>
            <div className="card">
                {/* Filters */}
                <div className="logs-header">
                    <h2>Inspection Logs</h2>
                    <div className="logs-filters">
                        <input className="input-filter" placeholder="Search label…"
                            value={label} onChange={(e) => setLabel(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleFilter()} />
                        <input className="input-filter input-small" type="number" placeholder="Min conf"
                            min={0} max={1} step={0.1} value={minConf} onChange={(e) => setMinConf(e.target.value)} />
                        <input className="input-filter input-small" type="number" placeholder="Max conf"
                            min={0} max={1} step={0.1} value={maxConf} onChange={(e) => setMaxConf(e.target.value)} />
                        <select className="input-filter" value={source} onChange={(e) => setSource(e.target.value)}>
                            <option value="">All sources</option>
                            <option value="image">Image</option>
                            <option value="video">Video</option>
                            <option value="webcam">Webcam</option>
                        </select>
                        <input className="input-filter" type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
                        <input className="input-filter" type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
                        <button className="btn btn-secondary" onClick={handleFilter}>🔍 Filter</button>
                        <button className="btn btn-ghost" onClick={clearFilters}>Clear</button>
                    </div>
                </div>

                {error && <div className="alert alert-error">⚠️ {error}</div>}

                {/* Table */}
                <div className="logs-table-wrap">
                    <table className="logs-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Image</th>
                                <th>Predicted Label</th>
                                <th>Confidence</th>
                                <th>Source</th>
                                <th>Latency</th>
                                <th>Date</th>
                            </tr>
                        </thead>
                        <tbody>
                            {loading && (
                                <tr><td colSpan={7} className="text-center"><span className="spinner" /></td></tr>
                            )}
                            {!loading && items.length === 0 && (
                                <tr><td colSpan={7} className="text-center text-muted">No records found</td></tr>
                            )}
                            {!loading && items.map((item) => (
                                <tr key={item.id}>
                                    <td>{item.id}</td>
                                    <td title={item.image_filename}>{truncate(item.image_filename, 20)}</td>
                                    <td title={item.predicted_label}>
                                        {truncate(item.predicted_label.replace(/--/g, ' › ').replace(/-/g, ' '), 30)}
                                    </td>
                                    <td>{confBadge(item.confidence)}</td>
                                    <td>{item.source_type}</td>
                                    <td>{item.latency_ms ? `${item.latency_ms.toFixed(1)}ms` : '—'}</td>
                                    <td>{new Date(item.created_at).toLocaleString()}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                <div className="logs-pagination">
                    <button className="btn btn-ghost" disabled={page <= 1}
                        onClick={() => fetchHistory(page - 1)}>← Previous</button>
                    <span className="pagination-info">
                        Page {page} of {totalPages} ({total} total)
                    </span>
                    <button className="btn btn-ghost" disabled={page >= totalPages}
                        onClick={() => fetchHistory(page + 1)}>Next →</button>
                </div>
            </div>
        </div>
    );
}
