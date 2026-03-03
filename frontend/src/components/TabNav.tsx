import type { TabId } from '../types';
import './TabNav.css';

interface TabNavProps {
    activeTab: TabId;
    onTabChange: (tab: TabId) => void;
}

const tabs: { id: TabId; icon: string; label: string }[] = [
    { id: 'upload', icon: '📸', label: 'Image Upload' },
    { id: 'video', icon: '🎬', label: 'Video Analysis' },
    { id: 'camera', icon: '📷', label: 'Live Camera' },
    { id: 'logs', icon: '📊', label: 'Inspection Logs' },
];

export default function TabNav({ activeTab, onTabChange }: TabNavProps) {
    return (
        <nav className="tab-nav" role="tablist">
            {tabs.map((t) => (
                <button
                    key={t.id}
                    role="tab"
                    aria-selected={activeTab === t.id}
                    className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
                    onClick={() => onTabChange(t.id)}
                >
                    <span className="tab-icon">{t.icon}</span>
                    {t.label}
                </button>
            ))}
        </nav>
    );
}
