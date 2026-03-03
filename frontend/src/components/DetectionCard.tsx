import type { Detection } from '../types';
import './DetectionCard.css';

interface DetectionCardProps {
    detection: Detection;
}

export default function DetectionCard({ detection }: DetectionCardProps) {
    const confPercent = (detection.confidence * 100).toFixed(1);
    const confClass = detection.confidence >= 0.8 ? 'high' : detection.confidence >= 0.5 ? 'medium' : 'low';
    const labelShort = detection.label.replace(/--/g, ' › ').replace(/-/g, ' ');

    return (
        <div className="detection-card">
            <div className="det-class-id">#{detection.class_id}</div>
            <div className="det-info">
                <div className="det-label" title={detection.label}>{labelShort}</div>
                <div className="det-sublabel">Class ID: {detection.class_id}</div>
            </div>
            <div className="confidence-bar">
                <div className={`confidence-fill ${confClass}`} style={{ width: `${confPercent}%` }} />
            </div>
            <div className={`confidence-text ${confClass}`}>{confPercent}%</div>
        </div>
    );
}
