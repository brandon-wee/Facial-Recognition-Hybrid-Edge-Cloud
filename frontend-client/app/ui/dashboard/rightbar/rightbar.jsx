"use client";

import styles from "./rightbar.module.css";
import { MdSecurity, MdRefresh, MdErrorOutline, MdCheckCircle, MdCloudDone, MdCloudOff } from "react-icons/md";
import { useNotifications } from "@/app/contexts/NotificationContext";
import { useState, useEffect } from "react";
import { getAllUsers, getAllUnknownPersons } from "@/app/lib/actions";
import { getApiStatus, getCloudApiUrl } from "@/app/lib/cloudApi";
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Rightbar = () => {
  const [systemStatus, setSystemStatus] = useState({
    cameras: "offline", // offline, online
    api: "error", // error, ok
    storage: 72, // percentage used
    apiDetails: null, // Additional API status details
  });
  
  const { addNotification, removeNotification } = useNotifications();
  const [detectionStats, setDetectionStats] = useState({
    today: { known: 0, unknown: 0 },
    week: { known: 0, unknown: 0 }
  });

  const [chartData, setChartData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCheckingApi, setIsCheckingApi] = useState(false);
  const [error, setError] = useState(null);
  
  // Check Cloud API status
  const checkApiStatus = async () => {
    setIsCheckingApi(true);
    try {
      // Remove previous API status notification
      removeNotification('api-status');
      
      const apiStatus = await getApiStatus();
      const isConnected = apiStatus.status === 'ok' || apiStatus.status === 'success';
      
      // Update system status with API status
      setSystemStatus(prev => ({
        ...prev,
        api: isConnected ? "ok" : "error",
        apiDetails: apiStatus,
      }));
      
      // Add new notification for API status
      addNotification({
        id: 'api-status',
        type: 'connection',
        status: isConnected ? 'connected' : 'error',
        message: isConnected 
          ? `API connected - ${getCloudApiUrl()}` 
          : `API error - ${apiStatus.message || 'Connection failed'}`,
        timestamp: new Date().toISOString()
      });
      
      return isConnected;
    } catch (error) {
      console.error("Error checking API status:", error);
      
      // Update status to disconnected
      setSystemStatus(prev => ({
        ...prev,
        api: "error",
        apiDetails: { status: 'error', message: error.message },
      }));
      
      // Add error notification
      addNotification({
        id: 'api-status',
        type: 'connection',
        status: 'error',
        message: `API connection failed: ${error.message}`,
        timestamp: new Date().toISOString()
      });
      
      return false;
    } finally {
      setIsCheckingApi(false);
    }
  };
  
  // Fetch detection stats from the database
  const fetchDetectionStats = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Check API status first
      await checkApiStatus();
      
      // Fetch all users and unknown persons
      const users = await getAllUsers();
      const unknownPersons = await getAllUnknownPersons();
      
      // Get today's date and beginning of the week
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      const weekStart = new Date(today);
      weekStart.setDate(today.getDate() - today.getDay()); // Start of the week (Sunday)
      
      // Filter for detections today and this week
      const knownToday = users.filter(user => 
        user.lastDetectedAt && new Date(user.lastDetectedAt) >= today
      ).length;
      
      const unknownToday = unknownPersons.filter(person => 
        person.lastDetectedAt && new Date(person.lastDetectedAt) >= today
      ).length;
      
      const knownWeek = users.filter(user => 
        user.lastDetectedAt && 
        new Date(user.lastDetectedAt) >= weekStart
      ).length;
      
      const unknownWeek = unknownPersons.filter(person => 
        person.lastDetectedAt && 
        new Date(person.lastDetectedAt) >= weekStart
      ).length;

      // Update detection stats
      setDetectionStats({
        today: { known: knownToday, unknown: unknownToday },
        week: { known: knownWeek, unknown: unknownWeek }
      });

      // Prepare data for the weekly chart
      const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const today_day = today.getDay();
      
      // Create arrays for the last 7 days
      const last7Days = Array(7).fill().map((_, i) => {
        const d = new Date(today);
        d.setDate(d.getDate() - 6 + i);
        return {
          date: d,
          label: days[(today_day - 6 + i + 7) % 7]
        };
      });
      
      // Count detections for each day
      const knownByDay = Array(7).fill(0);
      const unknownByDay = Array(7).fill(0);
      
      users.forEach(user => {
        if (user.lastDetectedAt) {
          const detectDate = new Date(user.lastDetectedAt);
          for (let i = 0; i < 7; i++) {
            if (detectDate >= last7Days[i].date && 
                detectDate < new Date(last7Days[i].date.getTime() + 24*60*60*1000)) {
              knownByDay[i]++;
              break;
            }
          }
        }
      });
      
      unknownPersons.forEach(person => {
        if (person.lastDetectedAt) {
          const detectDate = new Date(person.lastDetectedAt);
          for (let i = 0; i < 7; i++) {
            if (detectDate >= last7Days[i].date && 
                detectDate < new Date(last7Days[i].date.getTime() + 24*60*60*1000)) {
              unknownByDay[i]++;
              break;
            }
          }
        }
      });
      
      // Create chart data
      setChartData({
        labels: last7Days.map(d => d.label),
        datasets: [
          {
            label: 'Known',
            data: knownByDay,
            backgroundColor: 'rgba(85, 255, 85, 0.5)',
            borderColor: 'rgba(85, 255, 85, 1)',
            borderWidth: 1
          },
          {
            label: 'Unknown',
            data: unknownByDay,
            backgroundColor: 'rgba(255, 85, 85, 0.5)',
            borderColor: 'rgba(255, 85, 85, 1)',
            borderWidth: 1
          }
        ]
      });
      
    } catch (err) {
      console.error("Error fetching detection stats:", err);
      setError("Failed to fetch detection statistics");
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch data on initial load
  useEffect(() => {
    fetchDetectionStats();
    
    // Check API status periodically (every 60 seconds)
    const apiCheckInterval = setInterval(checkApiStatus, 60000);
    
    return () => clearInterval(apiCheckInterval);
  }, []);
  
  return (
    <div className={styles.container}>
      {/* System Status Card */}
      <div className={styles.item}>
        <div className={styles.header}>
          <h3 className={styles.title}>System Status</h3>
          <button 
            className={`${styles.refreshBtn} ${isCheckingApi ? styles.spinning : ''}`} 
            onClick={checkApiStatus}
            disabled={isCheckingApi}
            title="Check API status"
          >
            <MdRefresh />
          </button>
        </div>
        <div className={styles.content}>
          <div className={`${styles.statusItem} ${styles[systemStatus.api]}`}>
            <span className={styles.statusLabel}>Cloud API</span>
            <span className={styles.statusValue}>
              {systemStatus.api === "ok" ? (
                <><MdCloudDone className={styles.statusIcon} /> Connected</>
              ) : (
                <><MdCloudOff className={styles.statusIcon} /> Error</>
              )}
            </span>
          </div>
          <div className={styles.statusItem}>
            <span className={styles.statusLabel}>Storage</span>
            <div className={styles.progressBar}>
              <div 
                className={styles.progressFill} 
                style={{ 
                  width: `${systemStatus.storage}%`,
                  backgroundColor: systemStatus.storage > 90 
                    ? 'var(--error)' 
                    : systemStatus.storage > 70 
                      ? 'var(--warning)' 
                      : 'var(--success)'
                }}
              ></div>
            </div>
            <span className={styles.storageText}>{systemStatus.storage}% used</span>
          </div>
        </div>
      </div>
      
      {/* Detection Statistics */}
      <div className={styles.item}>
        <div className={styles.header}>
          <h3 className={styles.title}>Detection Stats</h3>
        </div>
        <div className={styles.statsContainer}>
          <div className={styles.statsPeriod}>
            <h4>Today</h4>
            <div className={styles.statRow}>
              <div className={styles.stat}>
                <span className={styles.statValue}>{detectionStats.today.known}</span>
                <span className={styles.statLabel}>Known</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>{detectionStats.today.unknown}</span>
                <span className={styles.statLabel}>Unknown</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>
                  {detectionStats.today.known + detectionStats.today.unknown}
                </span>
                <span className={styles.statLabel}>Total</span>
              </div>
            </div>
          </div>
          
          <div className={styles.statsPeriod}>
            <h4>This Week</h4>
            <div className={styles.statRow}>
              <div className={styles.stat}>
                <span className={styles.statValue}>{detectionStats.week.known}</span>
                <span className={styles.statLabel}>Known</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>{detectionStats.week.unknown}</span>
                <span className={styles.statLabel}>Unknown</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>
                  {detectionStats.week.known + detectionStats.week.unknown}
                </span>
                <span className={styles.statLabel}>Total</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Weekly Detection Chart */}
      <div className={styles.item}>
        <div className={styles.header}>
          <h3 className={styles.title}>Weekly Detections</h3>
        </div>
        <div className={styles.chartContainer}>
          {isLoading ? (
            <div className={styles.loadingChart}>Loading chart data...</div>
          ) : error ? (
            <div className={styles.errorChart}>{error}</div>
          ) : chartData ? (
            <Bar 
              data={chartData} 
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'top',
                    labels: {
                      color: 'var(--text)'
                    }
                  },
                  tooltip: {
                    backgroundColor: 'var(--bgSoft)',
                    titleColor: 'var(--text)',
                    bodyColor: 'var(--text)'
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    ticks: {
                      color: 'var(--textSoft)'
                    },
                    grid: {
                      color: 'var(--border)'
                    }
                  },
                  x: {
                    ticks: {
                      color: 'var(--textSoft)'
                    },
                    grid: {
                      color: 'var(--border)'
                    }
                  }
                }
              }}
            />
          ) : (
            <div className={styles.noDataChart}>No detection data available</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Rightbar;
