import React, { useRef, useEffect } from 'react';

export default function ChatConsole({ 
  chatHistory, 
  queryInput, 
  setQueryInput, 
  isAgentLoading, 
  agentCurrentThought, 
  agentThoughtsLog,
  onSendQuery 
}) {
  const messagesEndRef = useRef(null);

  const presetQueries = [
    { label: "Segment Base", text: "Segment customers based on balance and transaction frequency" },
    { label: "Check Regulars", text: "Which regular customers can be converted to priority? What should be done for the same?" },
    { label: "Average Size", text: "What is the average size of transactions for priority and regular customers?" },
    { label: "Explain rules", text: "On what basis were priority customers selected?" }
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isAgentLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!queryInput.trim() || isAgentLoading) return;
    onSendQuery(queryInput);
  };

  return (
    <div className="sidebar-panel">
      {/* Panel Header */}
      <div style={{
        padding: '24px',
        borderBottom: '1px solid var(--border-color)',
        background: 'rgba(10, 9, 21, 0.4)'
      }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '1.4rem',
          fontWeight: 700,
          background: 'var(--accent-gradient)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '4px'
        }}>
          Bank Personalization Agent
        </h1>
        <p style={{
          color: 'var(--text-secondary)',
          fontSize: '0.8rem',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            background: 'var(--color-success)',
            borderRadius: '50%',
            boxShadow: '0 0 8px var(--color-success)'
          }}></span>
          Ollama Llama3.2 Active & Online
        </p>
      </div>

      {/* Chat Messages Log */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {chatHistory.length === 0 ? (
          <div style={{
            textAlign: 'center',
            margin: 'auto 0',
            color: 'var(--text-secondary)',
            fontSize: '0.9rem',
            padding: '20px'
          }}>
            <div style={{
              fontSize: '2.5rem',
              marginBottom: '16px',
              opacity: 0.5
            }}>
              💬
            </div>
            Ask the bank agent to run segmentations, EDA queries, and detail explainability reports on customer data.
          </div>
        ) : (
          chatHistory.map((msg, index) => (
            <div 
              key={index}
              className="animate-fade-in"
              style={{
                alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '85%',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px'
              }}
            >
              {/* Message Sender Title */}
              <span style={{
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                fontWeight: 600,
                alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                {msg.sender === 'user' ? 'You' : 'Agent Analyst'}
              </span>

              {/* Message Body */}
              <div style={{
                background: msg.sender === 'user' ? 'var(--accent-gradient)' : 'rgba(255, 255, 255, 0.04)',
                border: msg.sender === 'user' ? 'none' : '1px solid var(--border-color)',
                color: '#fff',
                padding: '12px 16px',
                borderRadius: msg.sender === 'user' 
                  ? '16px 16px 4px 16px' 
                  : '16px 16px 16px 4px',
                fontSize: '0.9rem',
                whiteSpace: 'pre-line',
                boxShadow: msg.sender === 'user' ? '0 4px 12px rgba(139, 92, 246, 0.15)' : 'none',
                lineHeight: '1.5'
              }}>
                {msg.text}
              </div>

              {/* Expandable Agent Thought Logs */}
              {msg.thoughts && msg.thoughts.length > 0 && (
                <details style={{
                  fontSize: '0.75rem',
                  color: 'var(--text-secondary)',
                  marginTop: '4px',
                  background: 'rgba(0, 0, 0, 0.15)',
                  padding: '6px 10px',
                  borderRadius: '4px',
                  border: '1px solid rgba(255, 255, 255, 0.03)'
                }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, outline: 'none' }}>
                    View execution steps ({msg.thoughts.length})
                  </summary>
                  <ul style={{
                    listStyleType: 'none',
                    paddingTop: '6px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px',
                    borderLeft: '1px solid rgba(139, 92, 246, 0.3)',
                    paddingLeft: '10px',
                    marginLeft: '4px',
                    marginTop: '4px'
                  }}>
                    {msg.thoughts.map((th, tIdx) => (
                      <li key={tIdx} style={{ color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                        ✓ {th}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          ))
        )}

        {/* Live Loading Thoughts */}
        {isAgentLoading && (
          <div style={{
            alignSelf: 'flex-start',
            maxWidth: '85%',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            width: '100%'
          }}>
            <span style={{
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Agent Analyst
            </span>
            <div className="glass-card" style={{
              padding: '16px',
              borderRadius: '16px 16px 16px 4px',
              borderLeft: '3px solid var(--accent-primary)',
              animation: 'pulseGlow 2s infinite ease-in-out'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                <div style={{
                  width: '16px',
                  height: '16px',
                  border: '2px solid rgba(255,255,255,0.1)',
                  borderTopColor: 'var(--accent-primary)',
                  borderRadius: '50%',
                  animation: 'rotation 1s infinite linear'
                }}></div>
                <span style={{ fontSize: '0.85rem', color: '#fff', fontWeight: 600 }}>
                  Analyzing query...
                </span>
              </div>
              
              {/* Agent Thoughts logs rendering */}
              <div style={{
                fontFamily: 'monospace',
                fontSize: '0.75rem',
                color: 'var(--text-secondary)',
                maxHeight: '120px',
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
                background: 'rgba(0,0,0,0.2)',
                padding: '10px',
                borderRadius: '8px'
              }}>
                {agentThoughtsLog.map((log, lIdx) => (
                  <div key={lIdx} style={{ color: 'var(--text-secondary)' }}>
                    ✓ {log}
                  </div>
                ))}
                {agentCurrentThought && (
                  <div style={{ color: 'var(--accent-secondary)', fontWeight: 600 }}>
                    ❯ {agentCurrentThought}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Preset Action Chips */}
      <div style={{
        padding: '0 24px 12px',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '6px'
      }}>
        {presetQueries.map((chip, idx) => (
          <button
            key={idx}
            onClick={() => setQueryInput(chip.text)}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-secondary)',
              fontSize: '0.75rem',
              padding: '6px 12px',
              borderRadius: '9999px',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'all 0.2s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-primary)';
              e.currentTarget.style.color = '#fff';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-color)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }}
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Query Entry Input Box */}
      <form 
        onSubmit={handleSubmit}
        style={{
          padding: '16px 24px 24px',
          borderTop: '1px solid var(--border-color)',
          background: 'rgba(10, 9, 21, 0.4)'
        }}
      >
        <div style={{
          display: 'flex',
          gap: '8px',
          background: 'rgba(0, 0, 0, 0.3)',
          border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-sm)',
          padding: '4px 8px 4px 16px',
          alignItems: 'center'
        }}>
          <input
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Type query to segment, analyze..."
            disabled={isAgentLoading}
            style={{
              flex: 1,
              background: 'none',
              border: 'none',
              color: '#fff',
              fontSize: '0.9rem',
              outline: 'none',
              padding: '10px 0'
            }}
          />
          <button 
            type="submit"
            className="btn-primary"
            disabled={isAgentLoading || !queryInput.trim()}
            style={{
              padding: '8px 16px',
              fontSize: '0.85rem'
            }}
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
