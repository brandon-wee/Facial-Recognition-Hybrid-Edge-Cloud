"use client"

import styles from './chart.module.css'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';
import { useState, useEffect } from 'react';
import { getAllUsers } from '@/app/lib/actions';
import { MdInsights } from 'react-icons/md';

// Enhanced bar chart colors with better contrast and visual appeal
const BAR_COLORS = [
  "#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088fe", 
  "#00C49F", "#FFBB28", "#FF8042", "#a4de6c", "#d0ed57"
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className={styles.customTooltip}>
        <p className={styles.tooltipLabel}>{`${label}`}</p>
        <p className={styles.tooltipData}>{`Confidence: ${(payload[0].value * 100).toFixed(1)}%`}</p>
        <p className={styles.tooltipData}>{`Last detected: ${payload[0].payload.lastDetected}`}</p>
      </div>
    );
  }
  return null;
};

const Chart = () => {
  const [chartData, setChartData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchUserData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Get all users from database
        const users = await getAllUsers();
        
        // Filter to only include users who have been detected (have lastDetectedAt)
        const detectedUsers = users.filter(user => user.lastDetectedAt);
        
        // Format data for the chart
        const formattedData = detectedUsers.map(user => ({
          name: user.username,
          confidence: user.lastConfidence || 0,
          lastDetected: user.lastDetectedAt ? 
            new Date(user.lastDetectedAt).toLocaleString() : 'Never'
        }));
        
        // Sort by confidence level (highest first)
        formattedData.sort((a, b) => b.confidence - a.confidence);
        
        setChartData(formattedData);
      } catch (error) {
        console.error("Error fetching user data:", error);
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchUserData();
    
    // Refresh data every 30 seconds
    const intervalId = setInterval(fetchUserData, 30000);
    return () => clearInterval(intervalId);
  }, []);

  // Show a message when the component is loading or there's no data
  if (isLoading) {
    return <div className={styles.loadingContainer}>Loading user data...</div>
  }
  
  if (error) {
    return <div className={styles.errorContainer}>Error loading chart: {error}</div>
  }

  // If no data is available, display a message
  if (!chartData || chartData.length === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <h2 className={styles.title}>User Recognition Confidence</h2>
        </div>
        <div className={styles.noDataContainer}>
          <p>No detected users found.</p>
          <p>Data will appear here as users are identified by the system.</p>
        </div>
      </div>
    );
  }

  // If data is available, render the chart
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleContainer}>
          <MdInsights size={24} className={styles.titleIcon} />
          <div>
            <h2 className={styles.title}>User Recognition Confidence</h2>
            <p className={styles.subtitle}>
              Latest confidence levels for all detected users
            </p>
          </div>
        </div>
        <div className={styles.lastUpdated}>
          Updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
      
      <div className={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="name" 
              angle={-45} 
              textAnchor="end" 
              height={80} 
              tick={{fontSize: 13, fill: 'var(--textSoft)'}} 
              tickLine={{ stroke: 'var(--border)' }}
            />
            <YAxis 
              domain={[0, 1]} 
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              tick={{fill: 'var(--textSoft)'}}
              tickLine={{ stroke: 'var(--border)' }}
              axisLine={{ stroke: 'var(--border)' }}
              label={{ 
                value: 'Confidence Level', 
                angle: -90, 
                position: 'insideLeft',
                style: {fill: 'var(--textSoft)', fontSize: 13}
              }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
            <Legend 
              formatter={(value) => <span style={{color: 'var(--text)', fontSize: '14px'}}>{value}</span>}
              wrapperStyle={{ paddingTop: 15 }}
            />
            <Bar 
              dataKey="confidence" 
              name="Confidence Level" 
              radius={[6, 6, 0, 0]}
              animationDuration={1200}
              animationEasing="ease-in-out"
            >
              {chartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={BAR_COLORS[index % BAR_COLORS.length]} 
                  fillOpacity={0.9}
                  className={styles.chartBar}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Chart;