import { useState, useEffect } from 'react';
import Header from './components/Header';
import TabNav from './components/TabNav';
import ImageUpload from './components/ImageUpload';
import VideoAnalysis from './components/VideoAnalysis';
import LiveCamera from './components/LiveCamera';
import InspectionLogs from './components/InspectionLogs';
import { API_BASE } from './config';
import type { TabId } from './types';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('upload');
  const [apiStatus, setApiStatus] = useState<'connecting' | 'connected' | 'offline'>('connecting');

  useEffect(() => {
    checkApi();
  }, []);

  async function checkApi() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setApiStatus(res.ok ? 'connected' : 'offline');
    } catch {
      setApiStatus('offline');
    }
  }

  return (
    <div className="app">
      <Header status={apiStatus} />
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="main-content">
        {activeTab === 'upload' && <ImageUpload />}
        {activeTab === 'video' && <VideoAnalysis />}
        {activeTab === 'camera' && <LiveCamera />}
        {activeTab === 'logs' && <InspectionLogs />}
      </main>
    </div>
  );
}
