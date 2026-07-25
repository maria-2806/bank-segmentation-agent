import React from 'react';

export default function SegmentExplorer({ segmentationData, onDownloadCSV }) {
  if (!segmentationData) {
    return (
      <div style={{
        textAlign: 'center',
        padding: '60px',
        color: 'var(--text-secondary)'
      }}>
        <h2>No Customer Segmentation Data</h2>
        <p>Ask the agent to segment customers (e.g. <i>"Segment customers into 5 groups"</i>) to run clustering models and display rules.</p>
      </div>
    );
  }

  const { silhouette_score, cluster_sizes, profiles, rules, feature_importances } = segmentationData;

  // Segment Labels Helpers
  const getSegmentBadge = (id) => {
    // Standard mapping fallback if k=5, or just index
    const cleanId = parseInt(id);
    if (cleanId === 0) return <span className="badge badge-priority">Segment 0 (Priority)</span>;
    if (cleanId === 1) return <span className="badge badge-regular">Segment 1 (Regular)</span>;
    if (cleanId === 2) return <span className="badge badge-dormant">Segment 2 (Dormant)</span>;
    if (cleanId === 3) return <span className="badge badge-overleveraged">Segment 3 (Overleveraged)</span>;
    if (cleanId === 4) return <span className="badge badge-wealth">Segment 4 (Wealth/Investor)</span>;
    return <span className="badge badge-dormant">Cluster {cleanId}</span>;
  };

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', height: '100%', overflowY: 'auto' }}>
      
      {/* Top Banner stats */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: '#fff' }}>
            Machine Learning Cluster Profiles
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Discovered groups using K-Means Clustering on preprocessed customer parameters.
          </p>
        </div>
        
        <button className="btn-primary" onClick={onDownloadCSV}>
          📥 Download Segmented Customers (.CSV)
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Silhouette Score</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '4px' }}>
            <span style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--accent-secondary)' }}>
              {silhouette_score ? silhouette_score.toFixed(4) : 'N/A'}
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              (Density & Separation)
            </span>
          </div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Discovered Segments</span>
          <div style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: '#fff', marginTop: '4px' }}>
            {Object.keys(cluster_sizes).length} Clusters
          </div>
        </div>
      </div>

      {/* Grid: Rules/Profiles on Left, Feature Importance on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '20px' }}>
        
        {/* Global Cluster Profiles */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>
            Segment Profile Breakdown
          </h3>
          
          <div style={{ overflowX: 'auto' }}>
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Segment</th>
                  <th>Customer Count</th>
                  <th>Avg Balance</th>
                  <th>Avg Transactions</th>
                  <th>Avg Income</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(cluster_sizes).map((id) => {
                  const profile = profiles[id] || {};
                  return (
                    <tr key={id}>
                      <td>{getSegmentBadge(id)}</td>
                      <td style={{ fontWeight: 600 }}>{cluster_sizes[id]}</td>
                      <td style={{ color: 'var(--accent-primary)', fontWeight: 500 }}>
                        ${profile.account_balance ? Math.round(profile.account_balance).toLocaleString() : 'N/A'}
                      </td>
                      <td style={{ color: 'var(--accent-secondary)' }}>
                        {profile.transaction_frequency ? profile.transaction_frequency.toFixed(1) : 'N/A'}
                      </td>
                      <td>
                        ${profile.annual_income ? Math.round(profile.annual_income).toLocaleString() : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Feature Importance Card */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>
            Clustering Feature Importance
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            Statistical weights extracted from decision boundaries. Indicates which columns drive customer classification.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '8px' }}>
            {feature_importances && feature_importances.length > 0 ? (
              feature_importances.map((feat, idx) => (
                <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600 }}>
                    <span style={{ color: 'var(--text-primary)', fontFamily: 'monospace' }}>
                      {feat.feature}
                    </span>
                    <span style={{ color: 'var(--accent-secondary)' }}>
                      {(feat.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  {/* Progress bar background */}
                  <div style={{
                    width: '100%',
                    height: '8px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    {/* Filled bar width */}
                    <div style={{
                      width: `${feat.importance * 100}%`,
                      height: '100%',
                      background: 'var(--accent-gradient)',
                      borderRadius: '4px'
                    }} />
                  </div>
                </div>
              ))
            ) : (
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                No feature importances.
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Decision Tree Boundary Rules */}
      <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>
          Extracted Decision Tree Boundary Rules
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '-8px' }}>
          Decision tree criteria defining cluster assignment.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '8px' }}>
          {Object.keys(rules).map((segId) => (
            <div 
              key={segId}
              style={{
                background: 'rgba(0, 0, 0, 0.15)',
                border: '1px solid var(--border-color)',
                borderRadius: 'var(--radius-sm)',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                {getSegmentBadge(segId)}
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {rules[segId].map((r, rIdx) => (
                  <div 
                    key={rIdx}
                    style={{
                      fontFamily: 'monospace',
                      fontSize: '0.8rem',
                      color: 'var(--accent-secondary)',
                      background: 'rgba(255, 255, 255, 0.02)',
                      padding: '8px 12px',
                      borderRadius: '4px',
                      borderLeft: '2px solid var(--accent-secondary)',
                      wordBreak: 'break-all'
                    }}
                  >
                    IF {r.rule.replace(/ AND /g, '\n   AND ')}
                    <span style={{ color: 'var(--text-muted)', marginLeft: '8px' }}>
                      ({r.samples} samples)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
