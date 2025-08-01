import styles from "@/app/ui/dashboard/analytics/edge/edge.module.css";

const EdgeDevicesPage = () => {
  // Sample edge device data
  const edgeDevices = [
    { id: 1, name: "Edge Node 1", status: "Online", location: "Factory A", lastSeen: "Just now" },
    { id: 2, name: "Edge Node 2", status: "Online", location: "Factory B", lastSeen: "5 min ago" },
    { id: 3, name: "Edge Node 3", status: "Offline", location: "Factory A", lastSeen: "2 hours ago" },
    { id: 4, name: "Edge Node 4", status: "Online", location: "Factory C", lastSeen: "Just now" },
  ];

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>Edge Devices</h1>
        <button className={styles.addButton}>Add New Device</button>
      </div>

      <div className={styles.deviceGrid}>
        {edgeDevices.map((device) => (
          <div key={device.id} className={styles.deviceCard}>
            <div className={styles.deviceHeader}>
              <h2>{device.name}</h2>
              <span className={`${styles.status} ${device.status === "Online" ? styles.online : styles.offline}`}>
                {device.status}
              </span>
            </div>
            <div className={styles.deviceDetails}>
              <p><span>Location:</span> {device.location}</p>
              <p><span>Last Seen:</span> {device.lastSeen}</p>
            </div>
            <div className={styles.deviceActions}>
              <button className={styles.viewButton}>View Details</button>
              <button className={styles.configButton}>Configure</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EdgeDevicesPage;
