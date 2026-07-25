import React from 'react';

export default function SetupOverlay({ status, onRetry }) {
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      background: '#07060f',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      fontFamily: 'var(--font-primary)',
      padding: '24px'
    }}>
      <div className="glass-card animate-fade-in" style={{
        maxWidth: '560px',
        width: '100%',
        padding: '40px',
        textAlign: 'center',
        border: '1px solid rgba(239, 68, 68, 0.25)',
        boxShadow: '0 10px 40px rgba(239, 68, 68, 0.1)'
      }}>
        <div style={{
          width: '64px',
          height: '64px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 24px',
          color: '#ef4444',
          fontSize: '28px',
          fontWeight: 'bold'
        }}>
          !
        </div>
        
        <h2 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '1.8rem',
          marginBottom: '12px',
          color: '#f3f4f6'
        }}>
          Ollama Offline or Reachable Error
        </h2>
        
        <p style={{
          color: 'var(--text-secondary)',
          fontSize: '0.95rem',
          marginBottom: '24px',
          lineHeight: '1.6'
        }}>
          The banking segmentation agent requires a local instance of <strong>Ollama</strong> to be running in the background to handle natural language query processing.
        </p>

        <div style={{
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-sm)',
          padding: '20px',
          textAlign: 'left',
          marginBottom: '28px',
          fontSize: '0.85rem'
        }}>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: '10px', fontSize: '0.9rem' }}>
            How to resolve:
          </h4>
          <ol style={{ paddingLeft: '18px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <li>
              Make sure you have downloaded and installed Ollama from <a href="https://ollama.com" target="_blank" rel="noreferrer" style={{ color: 'var(--accent-secondary)', textDecoration: 'none', fontWeight: '600' }}>ollama.com</a>.
            </li>
            <li>
              Start the Ollama application on your system.
            </li>
            <li>
              Download/pull the required model using your terminal:
              <div style={{
                background: 'rgba(255, 255, 255, 0.05)',
                padding: '6px 12px',
                borderRadius: '4px',
                fontFamily: 'monospace',
                marginTop: '6px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                color: 'var(--accent-secondary)'
              }}>
                ollama pull llama3.2
              </div>
            </li>
          </ol>
        </div>

        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
          <button 
            className="btn-primary" 
            onClick={onRetry}
            style={{ padding: '12px 28px', fontSize: '0.95rem' }}
          >
            Check Connection
          </button>
          <a 
            href="https://ollama.com" 
            target="_blank" 
            rel="noreferrer"
            className="btn-secondary"
            style={{ padding: '12px 28px', fontSize: '0.95rem', textDecoration: 'none', display: 'flex', alignItems: 'center' }}
          >
            Download Ollama
          </a>
        </div>
      </div>
    </div>
  );
}
