import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar, Scatter } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);


export default function EdaView({ edaData }) {
  if (!edaData) {
    return (
      <div style={{
        textAlign: 'center',
        padding: '60px',
        color: 'var(--text-secondary)'
      }}>
        <h2>No Dataset Statistics Available</h2>
        <p>Run a query or load dataset statistics to populate EDA widgets.</p>
      </div>
    );
  }

  const { total_records, summary_statistics, chart_data } = edaData;
  const balanceHist = chart_data?.balance_histogram || [];
  const transHist = chart_data?.transaction_histogram || [];
  const incomeHist = chart_data?.income_histogram || [];
  const scatterPts = chart_data?.scatter_balance_vs_transactions || [];
  const axisTextColor = '#d1d5db';

  // 1. Balance Distribution Chart
  const balanceChartData = {
    labels: balanceHist.map(b => `${Math.round(b.bin_start/1000)}k`),
    datasets: [{
      label: 'Number of Customers',
      data: balanceHist.map(b => b.count),
      backgroundColor: 'rgba(139, 92, 246, 0.6)',
      borderColor: 'rgba(139, 92, 246, 1)',
      borderWidth: 1,
      borderRadius: 4
    }]
  };

  // 2. Transaction Frequency Chart
  const transChartData = {
    labels: transHist.map(t => Math.round(t.bin_start)),
    datasets: [{
      label: 'Number of Customers',
      data: transHist.map(t => t.count),
      backgroundColor: 'rgba(6, 182, 212, 0.6)',
      borderColor: 'rgba(6, 182, 212, 1)',
      borderWidth: 1,
      borderRadius: 4
    }]
  };

  // 3. Scatter Plot: Balance vs Transaction Frequency
  const scatterChartData = {
    datasets: [{
      label: 'Customers',
      data: scatterPts.map(pt => ({
        x: pt.account_balance,
        y: pt.transaction_frequency,
        id: pt.customer_id
      })),
      backgroundColor: 'rgba(236, 72, 153, 0.6)',
      pointRadius: 4,
      pointHoverRadius: 6
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#121024',
        titleColor: '#fff',
        bodyColor: '#e5e7eb',
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: axisTextColor, font: { size: 10 } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: axisTextColor, font: { size: 10 } }
      }
    }
  };

  const scatterOptions = {
    ...chartOptions,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (context) => {
            const pt = scatterPts[context.dataIndex];
            return `CUST: ${pt?.customer_id || ''} | Bal: $${context.raw.x.toLocaleString()} | Trans: ${context.raw.y}`;
          }
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: 'Account Balance ($)', color: axisTextColor },
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: axisTextColor }
      },
      y: {
        title: { display: true, text: 'Monthly Transactions', color: axisTextColor },
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: axisTextColor }
      }
    }
  };

  // Helper stats extraction
  const avgBalance = summary_statistics?.account_balance?.mean || 0;
  const avgIncome = summary_statistics?.annual_income?.mean || 0;
  const avgTrans = summary_statistics?.transaction_frequency?.mean || 0;
  const avgCreditScore = summary_statistics?.credit_score?.mean || 0;

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', height: '100%', overflowY: 'auto' }}>
      
      {/* Overview Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Total Records</span>
          <span style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: '#fff' }}>{total_records.toLocaleString()}</span>
        </div>

        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Avg Account Balance</span>
          <span style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--accent-primary)' }}>${Math.round(avgBalance).toLocaleString()}</span>
        </div>

        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Avg Monthly Transactions</span>
          <span style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--accent-secondary)' }}>{avgTrans.toFixed(1)}</span>
        </div>

        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>Avg Credit Score</span>
          <span style={{ fontSize: '1.8rem', fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--color-success)' }}>{Math.round(avgCreditScore)}</span>
        </div>

      </div>

      {/* Visualizations Panel Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px' }}>
        
        {/* Balance Histogram */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>Account Balance Distribution</h3>
          <div style={{ height: '240px', position: 'relative' }}>
            <Bar data={balanceChartData} options={chartOptions} />
          </div>
        </div>

        {/* Transactions Histogram */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>Monthly Transactions Distribution</h3>
          <div style={{ height: '240px', position: 'relative' }}>
            <Bar data={transChartData} options={chartOptions} />
          </div>
        </div>

        {/* Scatter Plot: Balance vs Transactions */}
        <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px', gridColumn: 'span 2' }}>
          <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', color: '#fff' }}>Behavioral Mapping: Balance vs Transaction Activity</h3>
          <div style={{ height: '320px', position: 'relative' }}>
            <Scatter data={scatterChartData} options={scatterOptions} />
          </div>
        </div>

      </div>
    </div>
  );
}
