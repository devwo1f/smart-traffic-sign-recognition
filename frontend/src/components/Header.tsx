import './Header.css';

interface HeaderProps {
    status: 'connecting' | 'connected' | 'offline';
}

const statusLabel: Record<string, string> = {
    connecting: 'Connecting…',
    connected: 'API Connected',
    offline: 'API Offline',
};

export default function Header({ status }: HeaderProps) {
    return (
        <header className="header">
            <div className="header-brand">
                <span className="logo">🚦</span>
                <div>
                    <h1 className="header-title">Traffic Sign Recognition</h1>
                    <p className="header-subtitle">AI-Powered Detection &amp; Classification</p>
                </div>
            </div>
            <div className={`status-indicator status-${status}`}>
                <span className="status-dot" />
                <span>{statusLabel[status]}</span>
            </div>
        </header>
    );
}
