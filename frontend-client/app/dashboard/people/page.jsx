"use client";

import styles from "@/app/ui/dashboard/people/people.module.css";
import Image from "next/image";
import { useState, useEffect } from "react";
import { MdRefresh, MdSearchOff, MdAdminPanelSettings, MdPersonOff } from "react-icons/md";
import { getAllUnknownPersons, getAllUsers } from "@/app/lib/actions";

// Enable debug mode
const DEBUG = true;

const PeoplePage = () => {
  const [statusFilter, setStatusFilter] = useState("all");
  const [detectedPeople, setDetectedPeople] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [debugInfo, setDebugInfo] = useState({
    knownUsersCount: 0,
    unknownPersonsCount: 0,
    error: null
  });
  
  const fetchDatabaseData = async () => {
    setIsLoading(true);
    try {
      console.log("Fetching data from database...");
      
      // Fetch both known users and unknown persons directly
      const knownUsers = await getAllUsers();
      const unknownPersons = await getAllUnknownPersons();
      
      // Filter known users to only those who have been detected
      const detectedUsers = knownUsers.filter(user => user.lastDetectedAt);
      console.log(`Filtered to ${detectedUsers.length} detected users out of ${knownUsers.length} total users`);
      
      // Safely map detected user data
      const knownUserObjects = Array.isArray(detectedUsers) ? detectedUsers.map(user => ({
        id: String(user._id || Math.random()),
        name: String(user.username || "Unnamed User"),
        isAdmin: Boolean(user.isAdmin),
        isActive: Boolean(user.isActive),
        image: user.img || "/noavatar.png",
        status: "known",
        confidenceScore: user.lastConfidence ? Number(user.lastConfidence) * 100 : 0,
        lastDetected: new Date(user.lastDetectedAt).toLocaleString()
      })) : [];
      
      // Process unknown persons, using standard avatar for all of them
      const unknownPeopleObjects = Array.isArray(unknownPersons) ? unknownPersons.map(person => ({
        id: person.unknownId || `unknown-${Math.random()}`,
        name: person.name || "Unknown Person",
        isAdmin: false,
        isActive: false,
        image: "/noavatar.png", // Always use default avatar for unknown
        status: "unknown",
        confidenceScore: person.lastConfidence ? person.lastConfidence * 100 : 0,
        lastDetected: person.lastDetectedAt ? 
                    new Date(person.lastDetectedAt).toLocaleString() : 'Never'
      })) : [];
      
      // Combine all people and sort by last detected time
      const allPeople = [...knownUserObjects, ...unknownPeopleObjects];
      console.log(`Total people to display: ${allPeople.length}`);
      
      // Sort by detection time
      const sortedPeople = allPeople.sort((a, b) => {
        if (a.lastDetected === 'Never') return 1;
        if (b.lastDetected === 'Never') return -1;
        return new Date(b.lastDetected) - new Date(a.lastDetected);
      });
      
      // Update state
      setDetectedPeople(sortedPeople);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Error fetching database data:", error);
      setDebugInfo(prev => ({
        ...prev,
        error: error.message
      }));
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch data on component mount
  useEffect(() => {
    fetchDatabaseData();
    // No auto refresh, only manual
  }, []);
  
  // Filter people based on status
  const filteredPeople = detectedPeople.filter(person => 
    statusFilter === "all" || person.status === statusFilter
  );

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>Detected Users & Unknown Persons</h1>
        <div className={styles.actions}>
          <button 
            className={styles.refreshButton}
            onClick={fetchDatabaseData}
            disabled={isLoading}
          >
            <MdRefresh className={isLoading ? styles.spinning : ""} /> Refresh
          </button>
        </div>
      </div>

      <div className={styles.filters}>
        <select 
          className={styles.filter}
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="all">All Detections</option>
          <option value="known">Known Users</option>
          <option value="unknown">Unknown Persons</option>
        </select>
        
        {lastUpdated && (
          <span className={styles.lastUpdated}>
            Last updated: {lastUpdated.toLocaleTimeString()} | 
            Found: {detectedPeople.length} people
          </span>
        )}
      </div>
      
      {isLoading ? (
        <div className={styles.loadingIndicator}>Loading user data...</div>
      ) : debugInfo.error ? (
        <div className={styles.errorContainer}>
          <h3>Error loading data</h3>
          <p>{debugInfo.error}</p>
        </div>
      ) : detectedPeople.length === 0 ? (
        <div className={styles.debugSection}>
          <div className={styles.noResults}>
            <MdSearchOff size={48} />
            <p>No users found in the database</p>
          </div>
        </div>
      ) : filteredPeople.length === 0 ? (
        <div className={styles.noResults}>
          <MdSearchOff size={48} />
          <p>No users match your filter: {statusFilter}</p>
        </div>
      ) : (
        filteredPeople.length > 0 && (
          <table className={styles.table}>
            <thead>
              <tr>
                <td>Person</td>
                <td>Status</td>
                <td>Admin</td>
                <td>Active</td>
                <td>Confidence</td>
                <td>Last Detected</td>
              </tr>
            </thead>
            <tbody>
              {filteredPeople.map((person) => (
                <tr key={person.id} className={styles[person.status]}>
                  <td>
                    <div className={styles.person}>
                      <Image
                        src={person.image}
                        alt={person.name}
                        width={35}
                        height={35}
                        className={styles.personImage}
                      />
                      {person.name}
                    </div>
                  </td>
                  <td>
                    <span className={`${styles.statusBadge} ${styles[person.status]}`}>
                      {person.status === "known" ? "Known" : "Unknown"}
                    </span>
                  </td>
                  <td>
                    {person.isAdmin ? (
                      <div className={styles.adminBadge}>
                        <MdAdminPanelSettings size={18} /> Admin
                      </div>
                    ) : (
                      "No"
                    )}
                  </td>
                  <td>
                    {person.isActive ? (
                      <span className={styles.activeBadge}>Active</span>
                    ) : (
                      <span className={styles.inactiveBadge}>
                        <MdPersonOff size={16} /> Inactive
                      </span>
                    )}
                  </td>
                  <td>{person.lastDetected === 'Never' ? 'N/A' : `${person.confidenceScore.toFixed(1)}%`}</td>
                  <td>{person.lastDetected}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      )}
    </div>
  );
};

export default PeoplePage;