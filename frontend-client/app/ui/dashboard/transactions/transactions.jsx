"use client";

import Image from "next/image";
import styles from "./transactions.module.css";
import { useState, useEffect } from "react";
import { getAllUnknownPersons, getAllUsers } from "@/app/lib/actions";
import { useNotifications } from "@/app/contexts/NotificationContext";

const Detections = () => {
  const [animate, setAnimate] = useState(false);
  const [recentDetections, setRecentDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const { addNotification } = useNotifications();
  
  useEffect(() => {
    setAnimate(true);
    fetchRecentDetections();
  }, []);

  const fetchRecentDetections = async () => {
    setIsLoading(true);
    try {
      // Fetch both known users and unknown persons
      const knownUsers = await getAllUsers();
      const unknownPersons = await getAllUnknownPersons();
      
      // Filter known users to only those who have been detected
      const detectedUsers = knownUsers.filter(user => user.lastDetectedAt);
      
      // Map detected users to our format
      const knownUserObjects = detectedUsers.map(user => ({
        id: String(user._id || Math.random()),
        name: String(user.username || "Unnamed User"),
        status: "verified",
        timestamp: new Date(user.lastDetectedAt).toLocaleString('en-US', {
          hour: 'numeric',
          minute: 'numeric',
          hour12: true,
          month: 'short',
          day: 'numeric'
        }),
        confidence: user.lastConfidence ? Number(user.lastConfidence) * 100 : 0,
        img: user.img || "/noavatar.png",
        lastDetectedAt: new Date(user.lastDetectedAt)
      }));
      
      // Map unknown persons
      const unknownPeopleObjects = unknownPersons.map(person => ({
        id: person.unknownId || `unknown-${Math.random()}`,
        name: person.name || "Unknown Person",
        status: "unverified",
        timestamp: person.lastDetectedAt ? 
          new Date(person.lastDetectedAt).toLocaleString('en-US', {
            hour: 'numeric',
            minute: 'numeric',
            hour12: true,
            month: 'short',
            day: 'numeric'
          }) : 'Never',
        confidence: person.lastConfidence ? person.lastConfidence * 100 : 0,
        img: "/noavatar.png",
        lastDetectedAt: person.lastDetectedAt ? new Date(person.lastDetectedAt) : null
      }));
      
      // Combine all people
      const allPeople = [...knownUserObjects, ...unknownPeopleObjects];
      
      // Sort by detection time (most recent first) and take top 5
      const sortedPeople = allPeople
        .filter(p => p.lastDetectedAt) // Ensure only detected people are included
        .sort((a, b) => b.lastDetectedAt - a.lastDetectedAt)
        .slice(0, 5);
      
      setRecentDetections(sortedPeople);

      // Add notification for demonstration
      if (sortedPeople.length > 0) {
        const mostRecent = sortedPeople[0];
        addNotification({
          id: Date.now(),
          type: 'detection',
          status: mostRecent.status === 'verified' ? 'known' : 'unknown',
          message: `${mostRecent.name} was detected recently`,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      console.error("Error fetching recent detections:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`${styles.container} ${animate ? styles.animate : ''}`}>
      <div className={styles.header}>
        <h2 className={styles.title}>Latest Person Detections</h2>
        <button className={styles.viewAll} onClick={fetchRecentDetections}>Refresh</button>
      </div>
      <div className={styles.tableWrapper}>
        {isLoading ? (
          <div className={styles.loading}>Loading recent detections...</div>
        ) : recentDetections.length === 0 ? (
          <div className={styles.noData}>No recent detections found</div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <td>Person</td>
                <td>Status</td>
                <td>Timestamp</td>
                <td>Confidence</td>
              </tr>
            </thead>
            <tbody>
              {recentDetections.map((detection, index) => (
                <tr key={detection.id} style={{animationDelay: `${index * 0.1}s`}}>
                  <td>
                    <div className={styles.user}>
                      <div className={styles.imgContainer}>
                        <Image
                          src={detection.img}
                          alt=""
                          width={40}
                          height={40}
                          className={styles.userImage}
                        />
                      </div>
                      <span className={styles.userName}>{detection.name}</span>
                    </div>
                  </td>
                  <td>
                    <span className={`${styles.status} ${styles[detection.status]}`}>
                      {detection.status === "unverified" && "Unverified"}
                      {detection.status === "verified" && "Verified"}
                      {detection.status === "falseAlarm" && "False Alarm"}
                    </span>
                  </td>
                  <td>
                    <span className={styles.timestamp}>{detection.timestamp}</span>
                  </td>
                  <td>
                    <div className={styles.confidenceWrapper}>
                      <div className={styles.confidenceBar}>
                        <div 
                          className={styles.confidenceFill} 
                          style={{
                            width: `${detection.confidence}%`,
                            backgroundColor: 
                              detection.confidence > 90 ? "var(--success)" : 
                              detection.confidence > 80 ? "var(--warning)" : 
                              "var(--error)"
                          }}
                        ></div>
                      </div>
                      <span className={styles.confidenceText}>{detection.confidence.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Detections;
