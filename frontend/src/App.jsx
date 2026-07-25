import React, { useState, useEffect } from 'react';
import SetupOverlay from './components/SetupOverlay';
import ChatConsole from './components/ChatConsole';
import EdaView from './components/EdaView';
import SegmentExplorer from './components/SegmentExplorer';
import CustomerCard from './components/CustomerCard';

export default function App() {
  const [activeTab, setActiveTab] = useState('eda');
  const [ollamaStatus, setOllamaStatus] = useState({ status: 'unknown' });
  const [isStatusLoading, setIsStatusLoading] = useState(true);
  
  // Chat console states
  const [chatHistory, setChatHistory] = useState([]);
  const [queryInput, setQueryInput] = useState('');
  const [isAgentLoading, setIsAgentLoading] = useState(false);
  const [agentCurrentThought, setAgentCurrentThought] = useState('');
  const [agentThoughtsLog, setAgentThoughtsLog] = useState([]);
  
  // Dashboard states
  const [edaData, setEdaData] = useState(null);
  const [segmentationData, setSegmentationData] = useState(null);
  const [campaignData, setCampaignData] = useState(null);
  
  // Customer explorer states
  const [searchedProfile, setSearchedProfile] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState('');

  // 1. Fetch startup status & default statistics
  const checkStatusAndLoadData = async () => {
    setIsStatusLoading(true);
    try {
      const statusRes = await fetch('/api/status');
      const statusData = await statusRes.json();
      setOllamaStatus(statusData.ollama || { status: 'offline' });
      
      if (statusData.ollama?.status === 'online') {
        // Load default statistics
        const dataRes = await fetch('/api/data');
        if (dataRes.ok) {
          const stats = await dataRes.json();
          setEdaData(stats);
        }
      }
    } catch (err) {
      setOllamaStatus({ status: 'offline', error: err.message });
    } finally {
      setIsStatusLoading(false);
    }
  };

  useEffect(() => {
    checkStatusAndLoadData();
  }, []);

  // 2. Query execution pipeline handler
  const handleSendQuery = async (queryText) => {
    setQueryInput('');
    setChatHistory(prev => [...prev, { sender: 'user', text: queryText }]);
    setIsAgentLoading(true);
    setAgentThoughtsLog([]);
    setAgentCurrentThought('Parsing query intent using local Llama3.2...');

    // Local thought flow simulation during blocking network request
    const thoughtsSequence = [
      'Identified request pattern. Scheduling pipeline tools...',
      'Accessing local SQLite/CSV transaction records...',
      'Computing feature metrics and data engineering matrices...',
      'Fitting scikit-learn models & extracting rules...'
    ];
    
    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < thoughtsSequence.length) {
        setAgentThoughtsLog(prev => [...prev, thoughtsSequence[stepIdx]]);
        setAgentCurrentThought(thoughtsSequence[stepIdx + 1] || 'Synthesizing report narrative...');
        stepIdx++;
      }
    }, 2800);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText })
      });
      
      clearInterval(interval);
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to get agent response');
      }

      const result = await res.json();
      
      // Update chat logs
      setChatHistory(prev => [...prev, {
        sender: 'agent',
        text: result.message,
        thoughts: result.thoughts || []
      }]);

      // Route data updates based on query classification
      const intent = result.intent;
      const data = result.data;

      if (intent === 'SEGMENTATION') {
        setSegmentationData(data);
        setActiveTab('segments');
      } else if (intent === 'EDA') {
        setEdaData(data);
        setActiveTab('eda');
      } else if (intent === 'EDA_AGG') {
        setActiveTab('eda');
      } else if (intent === 'EXPLAIN_SINGLE' || intent === 'RECOMMEND_SINGLE') {
        // Adapt schema from API to Customer Explorer
        setSearchedProfile({
          customer_id: data.customer_id,
          ...data.profile,
          segment_id: data.assigned_segment,
          explanation: {
            path: data.explanation_path
          },
          recommendations: data.recommendations,
          transition_steps: data.transition_steps
        });
        setActiveTab('customer');
      } else if (intent === 'RECOMMEND_GLOBAL') {
        setCampaignData(data);
        setActiveTab('campaigns');
      }

    } catch (err) {
      clearInterval(interval);
      setChatHistory(prev => [...prev, {
        sender: 'agent',
        text: `⚠️ Error executing query: ${err.message}. Make sure Ollama is active on port 11434.`
      }]);
    } finally {
      setIsAgentLoading(false);
      setAgentCurrentThought('');
    }
  };

  // 3. Customer explorer lookup helper
  const handleLookupCustomer = async (customerId) => {
    setIsSearching(true);
    setSearchError('');
    try {
      // We route lookups directly via the orchestrator by issuing an explain query
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: `Explain customer ${customerId.toUpperCase()}` })
      });
      
      if (!res.ok) {
        throw new Error(`Customer ${customerId} not found.`);
      }

      const result = await res.json();
      if (result.intent === 'EXPLAIN_SINGLE' && result.data) {
        const data = result.data;
        setSearchedProfile({
          customer_id: data.customer_id,
          ...data.profile,
          segment_id: data.assigned_segment,
          explanation: {
            path: data.explanation_path
          },
          recommendations: data.recommendations,
          transition_steps: data.transition_steps
        });
      } else {
        throw new Error(`Failed to retrieve profile trace for ${customerId}`);
      }
    } catch (err) {
      setSearchError(err.message);
    } finally {
      setIsSearching(false);
    }
  };

  const handleDownloadCSV = () => {
    window.open('/api/download', '_blank');
  };

  // Render Setup troubleshooting window if Ollama is unreachable
  if (!isStatusLoading && ollamaStatus.status !== 'online') {
    return <SetupOverlay status={ollamaStatus} onRetry={checkStatusAndLoadData} />;
  }

  return (
    <div className="app-container">
      
      {/* Left Column: Chat Assistant Console */}
      <ChatConsole 
        chatHistory={chatHistory}
        queryInput={queryInput}
        setQueryInput={setQueryInput}
        isAgentLoading={isAgentLoading}
        agentCurrentThought={agentCurrentThought}
        agentThoughtsLog={agentThoughtsLog}
        onSendQuery={handleSendQuery}
      />

      {/* Right Column: Interactive Dashboard Content */}
      <div className="dashboard-panel">
        
        {/* Navigation Tabs Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0 24px',
          borderBottom: '1px solid var(--border-color)',
          background: 'rgba(8, 7, 18, 0.5)',
          minHeight: '70px'
        }}>
          {/* Tab Navigation buttons */}
          <div style={{ display: 'flex', gap: '4px' }}>
            {[
              { id: 'eda', label: '📊 Data Insights' },
              { id: 'segments', label: '🤖 ML Segments' },
              { id: 'customer', label: '🔍 Customer Explorer' },
              { id: 'campaigns', label: '🎯 Campaigns' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  background: activeTab === tab.id ? 'rgba(255, 255, 255, 0.05)' : 'none',
                  border: 'none',
                  borderBottom: activeTab === tab.id ? '2px solid var(--accent-primary)' : '2px solid transparent',
                  color: activeTab === tab.id ? '#fff' : 'var(--text-secondary)',
                  padding: '12px 18px',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  outline: 'none'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Retail Banking Analytics Dashboard
          </div>
        </div>

        {/* Tab view panels */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {activeTab === 'eda' && <EdaView edaData={edaData} />}
          
          {activeTab === 'segments' && (
            <SegmentExplorer 
              segmentationData={segmentationData} 
              onDownloadCSV={handleDownloadCSV}
            />
          )}
          
          {activeTab === 'customer' && (
            <CustomerCard 
              onLookupCustomer={handleLookupCustomer}
              searchedProfile={searchedProfile}
              isSearching={isSearching}
              searchError={searchError}
            />
          )}

          {activeTab === 'campaigns' && (
            <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', height: '100%', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: '#fff' }}>
                    Segment Conversion Campaigns
                  </h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                    Outlining regular accounts ready to transition to priority customer standing.
                  </p>
                </div>
              </div>

              {campaignData ? (
                <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  
                  {/* Campaign summary stat */}
                  <div className="glass-card" style={{ padding: '20px' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Total Eligible Candidates</span>
                    <div style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--color-priority)', marginTop: '4px' }}>
                      {campaignData.total_eligible_candidates} Regular Customers
                    </div>
                  </div>

                  {/* Candidates table */}
                  <div className="glass-card" style={{ padding: '24px' }}>
                    <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff', marginBottom: '16px' }}>
                      Top 10 Target Conversion Candidates
                    </h3>
                    
                    <div style={{ overflowX: 'auto' }}>
                      <table className="custom-table">
                        <thead>
                          <tr>
                            <th>Customer ID</th>
                            <th>Account Balance</th>
                            <th>Transactions / mo</th>
                            <th>Annual Income</th>
                            <th>Credit Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {campaignData.candidates.map((cand) => (
                            <tr key={cand.customer_id}>
                              <td style={{ fontWeight: 'bold', color: 'var(--accent-secondary)' }}>{cand.customer_id}</td>
                              <td style={{ fontWeight: 600, color: '#fff' }}>${cand.account_balance.toLocaleString()}</td>
                              <td>{cand.transaction_frequency}</td>
                              <td>${cand.annual_income.toLocaleString()}</td>
                              <td>
                                <span className="badge" style={{
                                  background: cand.credit_score >= 700 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                                  color: cand.credit_score >= 700 ? 'var(--color-success)' : 'var(--color-warning)'
                                }}>
                                  {cand.credit_score}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                </div>
              ) : (
                <div className="glass-card" style={{
                  textAlign: 'center',
                  padding: '60px',
                  color: 'var(--text-secondary)'
                }}>
                  🎯 Run a recommendation campaign query (e.g. <i>"Which regular customers can be converted?"</i>) to populate target lead distributions.
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
