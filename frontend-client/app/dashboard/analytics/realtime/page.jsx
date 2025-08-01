"use client";

import { useState, useEffect, useRef } from "react";
import styles from "@/app/ui/dashboard/analytics/realtime/realtime.module.css";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { 
  MdPeopleAlt, MdSpeed, MdErrorOutline, MdFace, MdSwapHoriz, 
  MdHistory, MdDelete, MdDeleteSweep, MdRefresh, MdTimeline,
  MdZoomIn, MdVisibility, MdDevices
} from "react-icons/md";
import { getMetadata, getCloudApiUrl } from "@/app/lib/cloudApi";
import { updateUserConfidence, saveUnknownPerson, findUserByUsername } from "@/app/lib/actions";

// Debug mode to help troubleshoot API issues
const DEBUG = true;
const FETCH_INTERVAL = 5000; // 5 second interval for data fetching
const MAX_STORED_PERSONS = 50; // Maximum number of stored persons to avoid memory issues

const RealTimeDataPage = () => {
  const [metaData, setMetaData] = useState({ fps: 0, people_count: 0, names: [], best_sim: 0, raw: {} });
  const [fpsHistory, setFpsHistory] = useState([]);
  const [peopleCountHistory, setPeopleCountHistory] = useState([]);
  const [similarityHistory, setSimilarityHistory] = useState([]);
  const [detectedPersons, setDetectedPersons] = useState([]);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [apiStatus, setApiStatus] = useState({ 
    connected: false, 
    lastAttempt: null, 
    error: null,
    endpointChanged: false,
    previousEndpoint: null
  });
  const [refreshing, setRefreshing] = useState(false);
  
  const apiUrl = getCloudApiUrl();
  const prevApiUrlRef = useRef(apiUrl);

  // Track processed users to avoid duplicate processing
  const [processedUsers, setProcessedUsers] = useState(new Set());

  // Handle deleting a person from history
  const deletePerson = (personId) => {
    setDetectedPersons(prevPersons => 
      prevPersons.filter(person => person.id !== personId)
    );
    if (selectedPerson?.id === personId) {
      setSelectedPerson(null);
    }
  };

  // Clear all detection history
  const clearAllHistory = () => {
    setDetectedPersons([]);
    setSelectedPerson(null);
  };

  // Handle selecting a person for detailed view
  const handlePersonSelect = (person) => {
    setSelectedPerson(selectedPerson?.id === person.id ? null : person);
  };

  // Manual refresh function
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      const data = await getMetadata();
      setMetaData(data);
      
      // Process the data similar to the useEffect logic
      processMetadata(data);
      
      setApiStatus(prev => ({ 
        ...prev, 
        connected: true, 
        error: null,
        lastAttempt: new Date()
      }));
    } catch (error) {
      console.error("Manual refresh error:", error);
      setApiStatus(prev => ({
        ...prev,
        connected: false,
        error: error.message || "Failed to fetch metadata",
        lastAttempt: new Date()
      }));
    } finally {
      setRefreshing(false);
    }
  };

  // Process metadata and update confidence history
  const processMetadata = (data) => {
    if (!data.raw?.bboxes || !Array.isArray(data.raw.bboxes)) return;
    
    const timestamp = new Date();
    const timeStr = timestamp.toLocaleTimeString();
    
    // Process detected persons and update database
    if (data.raw.bboxes.length > 0) {
      const currentProcessed = new Set();
      
      // Update time series data for charts
      setFpsHistory(prev => {
        const newHistory = [...prev, { time: timeStr, fps: data.fps || 0 }];
        return newHistory.slice(-12); // Keep fewer data points for slower updates
      });
      
      setPeopleCountHistory(prev => {
        const newHistory = [...prev, { time: timeStr, count: data.people_count || 0 }];
        return newHistory.slice(-12);
      });
      
      setSimilarityHistory(prev => {
        const newHistory = [...prev, { time: timeStr, similarity: data.best_sim || 0 }];
        return newHistory.slice(-12);
      });
      
      // Process each detected person and update database
      data.raw.bboxes.forEach(async (person) => {
        const personKey = `${person.name}-${timestamp.getTime()}`;
        
        try {
          if (!person.name.startsWith("Unknown")) {
            // Known person - update in user database with confidence history
            const userData = await findUserByUsername(person.name);
            
            if (userData) {
              // Always update with current detection data - add to confidence history array
              await updateUserConfidence(person.name, person.similarity);
              
              if (DEBUG) {
                console.log(`[DETECTION] Updated known user ${person.name} with confidence ${person.similarity}`);
                console.log(`[DETECTION] Added entry to confidence history arrays`);
              }
            } else {
              console.warn(`User ${person.name} detected but not found in database`);
            }
            
            currentProcessed.add(personKey);
          } else {
            // Unknown person - save to unknown persons database with last confidence & timestamp
            await saveUnknownPerson({
              ...person,
              lastDetectedAt: timestamp
            });
            
            if (DEBUG) {
              console.log(`[DETECTION] Saved unknown person ${person.name} with confidence ${person.similarity}`);
            }
          }
        } catch (error) {
          console.error(`Failed to process person ${person.name}:`, error);
        }
      });
      
      // Update processed users set
      setProcessedUsers(currentProcessed);
      
      // Update the UI with detected persons
      setDetectedPersons(prevPersons => {
        // Add timestamp to each person detected in this frame
        const newPersons = data.raw.bboxes.map(person => ({
          ...person,
          detectedAt: timestamp.toLocaleTimeString(),
          detectionTime: timestamp,
          id: `${timestamp.getTime()}-${person.name}-${Math.random().toString(36).substring(2, 9)}`
        }));
        
        // Combine with previous persons and limit total number
        const combinedPersons = [...newPersons, ...prevPersons];
        return combinedPersons.slice(0, MAX_STORED_PERSONS);
      });
    }
  };

  // Fetch metadata at regular intervals
  useEffect(() => {
    let isMounted = true;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 3000;
    
    const fetchMetaData = async () => {
      try {
        // Check if API endpoint has changed
        const endpointChanged = prevApiUrlRef.current !== apiUrl;
        if (endpointChanged) {
          console.log(`[API ENDPOINT CHANGE] From ${prevApiUrlRef.current} to ${apiUrl}`);
        }
        
        // Update the ref with current URL
        prevApiUrlRef.current = apiUrl;
        
        if (isMounted) {
          setApiStatus(prev => ({ 
            ...prev, 
            lastAttempt: new Date(),
            endpointChanged: endpointChanged,
            previousEndpoint: endpointChanged ? prev.previousEndpoint || 'unknown' : prev.previousEndpoint
          }));
        }
        
        if (DEBUG) console.log(`Attempting to fetch metadata from: ${apiUrl}/metadata`);
        
        const data = await getMetadata();
        
        if (DEBUG) console.log("Received processed metadata:", data);
        
        // Reset retry count on success
        retryCount = 0;
        
        if (!isMounted) return;
        
        // Update metadata with processed information from cloud API
        setMetaData(data);
        
        // Process the metadata
        processMetadata(data);
        
        setApiStatus(prev => ({ 
          connected: true, 
          lastAttempt: new Date(), 
          error: null,
          endpointChanged: prev.endpointChanged,
          previousEndpoint: prev.previousEndpoint
        }));
      } catch (error) {
        console.error("Error fetching metadata:", error);
        
        if (!isMounted) return;
        
        // Add special handling for AbortError (timeout)
        const errorMessage = error.name === 'AbortError' 
          ? "Connection timed out. Please check your network or API server."
          : error.message || "Failed to fetch metadata";
          
        setApiStatus(prev => ({
          connected: false, 
          lastAttempt: new Date(), 
          error: errorMessage,
          endpointChanged: prev.endpointChanged,
          previousEndpoint: prev.previousEndpoint,
          retrying: retryCount < MAX_RETRIES
        }));
        
        // Auto-retry with exponential backoff if we haven't exceeded max retries
        if (retryCount < MAX_RETRIES && isMounted) {
          const delay = RETRY_DELAY * Math.pow(2, retryCount);
          console.log(`[RETRY] Will retry in ${delay/1000} seconds (attempt ${retryCount + 1}/${MAX_RETRIES})`);
          
          setTimeout(fetchMetaData, delay);
          retryCount++;
        }
      }
    };

    fetchMetaData(); // Initial fetch
    const intervalId = setInterval(fetchMetaData, FETCH_INTERVAL);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, [apiUrl]);

  return (
    <div className={styles.container}>
      {/* Header section with status */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>Real-time Edge Device Data</h1>
          <div className={styles.summary}>
            Monitor real-time person detection from connected edge devices
          </div>
        </div>
        
        <div className={styles.headerRight}>
          <button 
            className={`${styles.refreshButton} ${refreshing ? styles.refreshing : ''}`} 
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <MdRefresh className={refreshing ? styles.rotating : ''} /> 
            {refreshing ? 'Refreshing...' : 'Refresh Data'}
          </button>
          
          <div className={styles.apiStatus}>
            {apiStatus.connected ? (
              <div className={styles.statusIndicator}>
                <span className={styles.statusDot}></span>
                <span className={styles.connected}>
                  Connected to Cloud API {apiStatus.endpointChanged && (
                    <span className={styles.endpointChanged}>
                      <MdSwapHoriz /> API endpoint changed
                    </span>
                  )}
                </span>
              </div>
            ) : (
              <div className={styles.statusIndicator}>
                <span className={`${styles.statusDot} ${styles.statusError}`}></span>
                <span className={styles.disconnected}>
                  <MdErrorOutline /> API connection error: {apiStatus.error}
                  {apiStatus.retrying && (
                    <span className={styles.retrying}> (Retrying...)</span>
                  )}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats cards section */}
      <div className={styles.statsRow}>
        <div className={`${styles.statCard} ${styles.peopleCard}`}>
          <div className={styles.statIconWrapper}>
            <MdPeopleAlt className={styles.statIcon} />
          </div>
          <div className={styles.statContent}>
            <h3>People Count</h3>
            <span className={styles.statNumber}>{metaData.people_count || 0}</span>
          </div>
        </div>
        
        <div className={`${styles.statCard} ${styles.fpsCard}`}>
          <div className={styles.statIconWrapper}>
            <MdSpeed className={styles.statIcon} />
          </div>
          <div className={styles.statContent}>
            <h3>Current FPS</h3>
            <span className={styles.statNumber}>{metaData.fps?.toFixed(1) || 0}</span>
          </div>
        </div>
        
        <div className={`${styles.statCard} ${styles.confidenceCard}`}>
          <div className={styles.statIconWrapper}>
            <MdFace className={styles.statIcon} />
          </div>
          <div className={styles.statContent}>
            <h3>Best Confidence</h3>
            <span className={styles.statNumber}>{(metaData.best_sim * 100).toFixed(1) || 0}%</span>
          </div>
        </div>
        
        <div className={`${styles.statCard} ${styles.deviceCard}`}>
          <div className={styles.statIconWrapper}>
            <MdDevices className={styles.statIcon} />
          </div>
          <div className={styles.statContent}>
            <h3>Edge Devices</h3>
            <span className={styles.statNumber}>1</span>
          </div>
        </div>
      </div>
      
      {/* Main content area with tabs */}
      <div className={styles.mainContent}>
        <div className={styles.contentColumns}>
          {/* Left column - Current detection */}
          <div className={styles.leftColumn}>
            <div className={styles.sectionCard}>
              <div className={styles.sectionHeader}>
                <h2 className={styles.sectionTitle}>
                  <MdVisibility /> Current Detection
                </h2>
              </div>
              
              <div className={styles.personsContainer}>
                {(!metaData.raw?.bboxes || !Array.isArray(metaData.raw.bboxes) || metaData.raw.bboxes.length === 0) ? (
                  <div className={styles.noPersonsDetected}>
                    <MdPeopleAlt size={40} className={styles.emptyIcon} />
                    <p>No persons detected in the current frame</p>
                  </div>
                ) : (
                  <div className={styles.personsGrid}>
                    {metaData.raw.bboxes.map((face, index) => (
                      <div key={index} className={styles.personCard}>
                        {face.crop ? (
                          <img 
                            src={`data:image/png;base64,${face.crop}`} 
                            alt={`Person ${index + 1}`} 
                            className={styles.personImage} 
                          />
                        ) : (
                          <div className={styles.noImagePlaceholder}>No Image</div>
                        )}
                        <div className={styles.personDetails}>
                          <h3 className={`${styles.personName} ${face.name.startsWith("Unknown") ? styles.unknownPerson : ''}`}>
                            {face.name}
                          </h3>
                          <div className={styles.confidenceBar}>
                            <div 
                              className={`${styles.confidenceFill} ${face.similarity > 0.8 ? styles.highConfidence : face.similarity > 0.6 ? styles.mediumConfidence : styles.lowConfidence}`}
                              style={{width: `${(face.similarity * 100).toFixed(1)}%`}}
                            ></div>
                            <span className={styles.confidenceLabel}>
                              {(face.similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Person detail view when a person is selected */}
            {selectedPerson && (
              <div className={`${styles.sectionCard} ${styles.personDetail}`}>
                <div className={styles.sectionHeader}>
                  <h2 className={styles.sectionTitle}>
                    <MdZoomIn /> Person Details
                  </h2>
                  <button 
                    className={styles.closeButton}
                    onClick={() => setSelectedPerson(null)}
                  >
                    ×
                  </button>
                </div>
                
                <div className={styles.personDetailContent}>
                  <div className={styles.personDetailImage}>
                    {selectedPerson.crop ? (
                      <img 
                        src={`data:image/png;base64,${selectedPerson.crop}`} 
                        alt={`Person ${selectedPerson.name}`} 
                        className={styles.detailImage}
                      />
                    ) : (
                      <div className={styles.noDetailImage}>No Image Available</div>
                    )}
                  </div>
                  
                  <div className={styles.personDetailInfo}>
                    <h3 className={styles.detailName}>{selectedPerson.name}</h3>
                    <div className={styles.detailItems}>
                      <div className={styles.detailItem}>
                        <span className={styles.detailLabel}>Confidence:</span>
                        <span className={styles.detailValue}>{(selectedPerson.similarity * 100).toFixed(2)}%</span>
                      </div>
                      <div className={styles.detailItem}>
                        <span className={styles.detailLabel}>Detected at:</span>
                        <span className={styles.detailValue}>{selectedPerson.detectedAt}</span>
                      </div>
                      <div className={styles.detailItem}>
                        <span className={styles.detailLabel}>Position:</span>
                        <span className={styles.detailValue}>[{selectedPerson.bbox?.join(', ') || 'N/A'}]</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Right column - Detection History */}
          <div className={styles.rightColumn}>
            <div className={styles.sectionCard}>
              <div className={styles.sectionHeader}>
                <h2 className={styles.sectionTitle}>
                  <MdHistory /> Detection History
                </h2>
                {detectedPersons.length > 0 && (
                  <button 
                    className={styles.clearAllBtn} 
                    onClick={clearAllHistory}
                    title="Clear all detection history"
                  >
                    <MdDeleteSweep /> Clear All
                  </button>
                )}
              </div>
              
              <div className={styles.personsHistoryContainer}>
                {detectedPersons.length === 0 ? (
                  <div className={styles.noPersonsDetected}>
                    <MdHistory size={40} className={styles.emptyIcon} />
                    <p>No detection history available</p>
                  </div>
                ) : (
                  <div className={styles.personsList}>
                    {detectedPersons.map((person) => (
                      <div 
                        key={person.id} 
                        className={`${styles.personHistoryItem} ${selectedPerson?.id === person.id ? styles.selected : ''}`}
                        onClick={() => handlePersonSelect(person)}
                      >
                        <div className={styles.personThumbnail}>
                          {person.crop ? (
                            <img 
                              src={`data:image/png;base64,${person.crop}`} 
                              alt={`Person ${person.name}`} 
                              className={styles.personThumbnailImage}
                            />
                          ) : (
                            <div className={styles.noThumbnail}>No Image</div>
                          )}
                        </div>
                        <div className={styles.personHistoryDetails}>
                          <div className={`${styles.personHistoryName} ${person.name.startsWith("Unknown") ? styles.unknownPerson : ''}`}>
                            {person.name}
                          </div>
                          <div className={styles.personHistoryMeta}>
                            <span className={styles.personConfidence}>
                              {(person.similarity * 100).toFixed(1)}%
                            </span>
                            <span className={styles.personTime}>
                              {person.detectedAt}
                            </span>
                          </div>
                        </div>
                        <button 
                          className={styles.deletePersonBtn} 
                          onClick={(e) => {
                            e.stopPropagation();
                            deletePerson(person.id);
                          }}
                          title="Remove from history"
                        >
                          <MdDelete size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Charts section - enhanced look */}
      <div className={styles.chartsSection}>
        <div className={styles.chartsSectionHeader}>
          <h2><MdTimeline /> Real-Time Analytics</h2>
        </div>
        
        <div className={styles.chartsGrid}>
          {/* People Count Chart */}
          <div className={styles.chartCard}>
            <div className={styles.chartHeader}>
              <h3><MdPeopleAlt /> People Count</h3>
            </div>
            <div className={styles.chartContainer}>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart
                  data={peopleCountHistory}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                  <YAxis domain={[0, 'auto']} stroke="rgba(255,255,255,0.5)" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(0, 128, 128, 0.8)',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="count"
                    name="People" 
                    stroke="#00CED1" 
                    fill="rgba(0, 206, 209, 0.2)"
                    strokeWidth={2}
                    activeDot={{ r: 6, fill: "#00f7ff" }}
                    isAnimationActive={true}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Face Recognition Confidence Chart */}
          <div className={styles.chartCard}>
            <div className={styles.chartHeader}>
              <h3><MdFace /> Recognition Confidence</h3>
            </div>
            <div className={styles.chartContainer}>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart
                  data={similarityHistory}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                  <YAxis 
                    domain={[0, 1]} 
                    tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`}
                    stroke="rgba(255,255,255,0.5)"
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(0, 128, 128, 0.8)',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white'
                    }}
                    formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Confidence"]}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="similarity" 
                    name="Confidence" 
                    stroke="#FF8C00" 
                    fill="rgba(255, 140, 0, 0.2)"
                    strokeWidth={2}
                    activeDot={{ r: 6, fill: "#ffae42" }}
                    isAnimationActive={true}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* FPS Chart */}
          <div className={styles.chartCard}>
            <div className={styles.chartHeader}>
              <h3><MdSpeed /> Frames Per Second</h3>
            </div>
            <div className={styles.chartContainer}>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart
                  data={fpsHistory}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                  <YAxis domain={[0, 'auto']} stroke="rgba(255,255,255,0.5)" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(0, 128, 128, 0.8)',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="fps"
                    name="FPS" 
                    stroke="#20B2AA" 
                    fill="rgba(32, 178, 170, 0.2)"
                    strokeWidth={2}
                    activeDot={{ r: 6, fill: "#3dfcf0" }}
                    isAnimationActive={true}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeDataPage;
