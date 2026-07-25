import React, { useState } from 'react';

export default function CustomerCard({ 
  onLookupCustomer, 
  searchedProfile, 
  isSearching, 
  searchError 
}) {
  const [custIdInput, setCustIdInput] = useState('');

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (!custIdInput.trim() || isSearching) return;
    onLookupCustomer(custIdInput.trim());
  };

  const getSegmentName = (segId) => {
    const id = parseInt(segId);
    if (id === 0) return 'Priority';
    if (id === 1) return 'Regular';
    if (id === 2) return 'Dormant';
    if (id === 3) return 'Overleveraged';
    if (id === 4) return 'Wealth/Investor';
    return `Cluster ${id}`;
  };

  const getSegmentBadgeClass = (segId) => {
    const id = parseInt(segId);
    if (id === 0) return 'badge-priority';
    if (id === 1) return 'badge-regular';
    if (id === 2) return 'badge-dormant';
    if (id === 3) return 'badge-overleveraged';
    if (id === 4) return 'badge-wealth';
    return 'badge-dormant';
  };

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', height: '100%', overflowY: 'auto' }}>
      
      {/* Search Input Card */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem', color: '#fff', marginBottom: '12px' }}>
          Query Single Customer Records
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '20px' }}>
          Retrieve deep profiling, machine learning classification paths, and custom recommended marketing campaigns.
        </p>

        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: '12px' }}>
          <div style={{
            flex: 1,
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-sm)',
            padding: '4px 16px',
            display: 'flex',
            alignItems: 'center'
          }}>
            <input
              type="text"
              placeholder="Enter Customer ID (e.g. CUST0001, CUST0025)..."
              value={custIdInput}
              onChange={(e) => setCustIdInput(e.target.value)}
              disabled={isSearching}
              style={{
                width: '100%',
                background: 'none',
                border: 'none',
                color: '#fff',
                outline: 'none',
                fontSize: '0.9rem',
                padding: '10px 0'
              }}
            />
          </div>
          <button 
            type="submit" 
            className="btn-primary"
            disabled={isSearching || !custIdInput.trim()}
            style={{ padding: '0 28px' }}
          >
            {isSearching ? 'Loading...' : 'Search'}
          </button>
        </form>

        {searchError && (
          <div style={{ color: 'var(--color-danger)', fontSize: '0.85rem', marginTop: '12px', fontWeight: 600 }}>
            ⚠️ {searchError}
          </div>
        )}
      </div>

      {/* Searched Profile Display */}
      {searchedProfile ? (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Header Card */}
          <div className="glass-card" style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '4px' }}>
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.8rem', color: '#fff', fontWeight: 700 }}>
                  {searchedProfile.customer_id}
                </h2>
                <span className={`badge ${getSegmentBadgeClass(searchedProfile.segment_id)}`}>
                  {getSegmentName(searchedProfile.segment_id)} Segment
                </span>
                {searchedProfile.is_outlier === 1 && (
                  <span className="badge badge-overleveraged">Anomaly Outlier</span>
                )}
                {searchedProfile.is_boundary === 1 && (
                  <span className="badge badge-regular" style={{ borderColor: '#f59e0b', color: '#f59e0b', background: 'rgba(245,158,11,0.15)' }}>
                    Boundary Case
                  </span>
                )}
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                {searchedProfile.occupation} • {searchedProfile.region} Region • {searchedProfile.tenure_months} Months Member
              </p>
            </div>
            
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Account Balance</div>
              <div style={{ fontSize: '1.6rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--accent-secondary)' }}>
                ${parseFloat(searchedProfile.account_balance).toLocaleString()}
              </div>
            </div>
          </div>

          {/* Profile Metrics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
            
            {/* Demographics Card */}
            <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--text-primary)', textTransform: 'uppercase', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', fontWeight: 700 }}>
                Demographics
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.9rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Age</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{searchedProfile.age} years</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Occupation</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{searchedProfile.occupation}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Region</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{searchedProfile.region}</span>
                </div>
              </div>
            </div>

            {/* Financial Status Card */}
            <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--text-primary)', textTransform: 'uppercase', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', fontWeight: 700 }}>
                Financial Assets
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.9rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Annual Income</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>${parseFloat(searchedProfile.annual_income).toLocaleString()}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Credit Score</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{searchedProfile.credit_score}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Investment Bal</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>${parseFloat(searchedProfile.investment_balance).toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Account Behavior Card */}
            <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--text-primary)', textTransform: 'uppercase', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', fontWeight: 700 }}>
                Activity & Debt
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.9rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Transactions / mo</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{searchedProfile.transaction_frequency}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Avg Trans Amount</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>${searchedProfile.avg_transaction_amount}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Debt to Income</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{searchedProfile.debt_to_income}</span>
                </div>
              </div>
            </div>

          </div>

          {/* Explainability Decision Path & Rules */}
          {searchedProfile.explanation && (
            <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>
                Machine Learning Classification Explanation
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '-8px' }}>
                Traced decision path rules computed from our shallow decision tree boundaries:
              </p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {searchedProfile.explanation.path.map((step, idx) => (
                  <div 
                    key={idx}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '12px 16px',
                      fontSize: '0.85rem'
                    }}
                  >
                    <div style={{
                      width: '24px',
                      height: '24px',
                      background: 'rgba(139,92,246,0.15)',
                      border: '1px solid rgba(139,92,246,0.4)',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.75rem',
                      fontWeight: 'bold',
                      color: 'var(--accent-primary)'
                    }}>
                      {idx + 1}
                    </div>
                    <div style={{ flex: 1 }}>
                      Checking feature <strong style={{ fontFamily: 'monospace' }}>{step.feature}</strong>:
                      Current value <strong style={{ color: '#fff' }}>{step.raw_value.toLocaleString()}</strong> is 
                      <strong> {step.comparison} </strong> threshold boundary value 
                      <strong style={{ color: 'var(--accent-secondary)' }}> {step.threshold.toLocaleString()}</strong>.
                    </div>
                    <div style={{ color: 'var(--color-success)', fontWeight: 'bold' }}>
                      ✓ Match
                    </div>
                  </div>
                ))}
              </div>

              {/* Classification conclusion */}
              <div style={{
                background: 'rgba(16, 185, 129, 0.05)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                padding: '14px',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem',
                color: '#e5e7eb',
                lineHeight: '1.5'
              }}>
                <strong>Conclusion:</strong> Traced decision rules assign customer to Segment {searchedProfile.segment_id} (<strong>{getSegmentName(searchedProfile.segment_id)} Customer</strong>) with high statistical confidence.
              </div>
            </div>
          )}

          {/* Product Recommendations & Campaigns */}
          {searchedProfile.recommendations && (
            <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>
                Personalized Cross-Selling & Retention Campaigns
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
                {searchedProfile.recommendations.map((rec, idx) => (
                  <div 
                    key={idx}
                    style={{
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '16px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span className="badge badge-regular" style={{
                        background: rec.type === 'Up-Sell' ? 'rgba(139,92,246,0.15)' : 'rgba(6,182,212,0.15)',
                        color: rec.type === 'Up-Sell' ? 'var(--accent-primary)' : 'var(--accent-secondary)',
                        borderColor: rec.type === 'Up-Sell' ? 'rgba(139,92,246,0.3)' : 'rgba(6,182,212,0.3)'
                      }}>
                        {rec.type}
                      </span>
                    </div>
                    <h4 style={{ color: '#fff', fontSize: '1rem', fontWeight: 600 }}>{rec.product}</h4>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', lineHeight: '1.5' }}>{rec.reason}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Segment Upgrade Transition Path */}
          {searchedProfile.transition_steps && searchedProfile.transition_steps.length > 0 && (
            <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>
                Priority Status Transition Path Analysis
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '-8px' }}>
                To encourage this customer to upgrade to <strong>Priority Status (Segment 0)</strong>, they need to reach the following average segment boundaries:
              </p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {searchedProfile.transition_steps.map((step, idx) => (
                  <div 
                    key={idx}
                    style={{
                      background: 'rgba(217, 119, 6, 0.03)',
                      border: '1px solid rgba(217, 119, 6, 0.15)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '14px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                      fontSize: '0.85rem'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                      <span style={{ color: '#fff', fontFamily: 'monospace' }}>{step.metric.toUpperCase()}</span>
                      <span style={{ color: 'var(--color-priority)' }}>Gap: +{typeof step.difference === 'number' ? step.difference.toLocaleString(undefined, {maximumFractionDigits:1}) : step.difference}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '16px', color: 'var(--text-secondary)', fontSize: '0.8rem', margin: '4px 0' }}>
                      <span>Current: <strong>{step.current.toLocaleString(undefined, {maximumFractionDigits:1})}</strong></span>
                      <span>Target Average: <strong>{step.target.toLocaleString(undefined, {maximumFractionDigits:1})}</strong></span>
                    </div>
                    <div style={{ color: 'var(--text-primary)', fontSize: '0.82rem', borderTop: '1px solid rgba(217, 119, 6, 0.1)', paddingTop: '6px', marginTop: '2px' }}>
                      👉 <strong>Action:</strong> {step.action}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      ) : (
        <div className="glass-card" style={{
          textAlign: 'center',
          padding: '60px',
          color: 'var(--text-secondary)',
          margin: 'auto 0'
        }}>
          🔍 Search for a customer ID to display their personal assets, transaction metrics, explainability decisions, and product campaigns.
        </div>
      )}

    </div>
  );
}
