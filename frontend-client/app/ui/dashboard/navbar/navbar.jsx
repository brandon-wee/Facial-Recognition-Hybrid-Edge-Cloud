"use client";
import { usePathname } from "next/navigation";
import styles from "./navbar.module.css";
import {
  MdNotifications,
  MdOutlineDarkMode,
  MdOutlineLightMode,
  MdClose,
  MdCheckCircle,
  MdErrorOutline,
  MdFace,
  MdInfo,
} from "react-icons/md";
import { useTheme } from "@/app/contexts/ThemeContext";
import { useNotifications } from "@/app/contexts/NotificationContext";
import { useState, useEffect } from "react";


const Navbar = () => {
  const pathname = usePathname();
  const { isDarkMode, toggleTheme } = useTheme();
  const { notifications, removeNotification, addNotification } = useNotifications();
  const [isNotificationsPanelOpen, setIsNotificationsPanelOpen] = useState(false);

  // Use useEffect to verify the theme state when component mounts
  useEffect(() => {
    console.log("Current theme:", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);
  
  // Enhanced theme toggle with feedback
  const handleThemeToggle = () => {
    console.log("Theme toggle clicked, changing from", isDarkMode ? "dark" : "light");
    toggleTheme();
  };

  // Add API not connected notification
  useEffect(() => {
    // Add simple API not connected notification
    addNotification({
      id: 'api-status',
      type: 'connection',
      status: 'error',
      message: 'API is not connected',
      timestamp: new Date().toISOString()
    });
  }, [addNotification]);

  // Simulate unknown person detection for demonstration
  useEffect(() => {
    const timer = setTimeout(() => {
      addNotification({
        id: Date.now(),
        type: 'detection',
        status: 'unknown',
        message: 'Unknown person detected in video feed',
        timestamp: new Date().toISOString()
      });
    }, 3000);

    return () => clearTimeout(timer);
  }, [addNotification]);

  // Group notifications by type
  const apiNotifications = notifications.filter(n => n.type === 'connection');
  const detectionNotifications = notifications.filter(n => n.type === 'detection');
  const otherNotifications = notifications.filter(n => !n.type || (n.type !== 'connection' && n.type !== 'detection'));

  // Function to render notification icon based on type
  const renderNotificationIcon = (notif) => {
    if (notif.type === 'connection') {
      return notif.status === 'connected' ? 
        <MdCheckCircle className={styles.notifIconSuccess} /> : 
        <MdErrorOutline className={styles.notifIconError} />;
    } else if (notif.type === 'detection') {
      return <MdFace className={styles.notifIconWarning} />;
    } else {
      return <MdInfo className={styles.notifIconInfo} />;
    }
  };

  // Clean path name for title
  const getPageTitle = () => {
    const path = pathname.split("/").pop();
    // Convert to title case and replace hyphens/underscores with spaces
    return path ? path.charAt(0).toUpperCase() + path.slice(1).replace(/[-_]/g, ' ') : 'Dashboard';
  };

  return (
    <div className={styles.container}>
      <div className={styles.titleWrapper}>
        <h1 className={styles.title}>{getPageTitle()}</h1>
      </div>
      <div className={styles.menu}>
        <div className={styles.icons}>
          <div className={styles.iconWrapper}>
            <MdNotifications 
              size={22} 
              title="Notifications" 
              onClick={() => setIsNotificationsPanelOpen(!isNotificationsPanelOpen)}
              className={styles.icon}
            />
            {notifications.length > 0 && (
              <span className={styles.badge}>
                {notifications.length}
              </span>
            )}
            {isNotificationsPanelOpen && (
              <div className={styles.notificationsPanel}>
                <div className={styles.notificationHeader}>
                  <h3>Notifications</h3>
                  {notifications.length > 0 && (
                    <button 
                      className={styles.clearAllBtn}
                      onClick={() => notifications.forEach(n => removeNotification(n.id))}
                    >
                      Clear all
                    </button>
                  )}
                </div>
                
                {notifications.length === 0 ? (
                  <div className={styles.noNotifications}>
                    No notifications
                  </div>
                ) : (
                  <>
                    {apiNotifications.length > 0 && (
                      <div className={styles.notificationCategory}>
                        <h4>API Status</h4>
                        {apiNotifications.map((notif) => (
                          <div key={notif.id} className={`${styles.notification} ${styles[notif.status || 'default']}`}>
                            {renderNotificationIcon(notif)}
                            <div className={styles.notificationContent}>
                              <p>{notif.message}</p>
                              <small>{new Date(notif.timestamp).toLocaleTimeString()}</small>
                            </div>
                            <MdClose 
                              onClick={() => removeNotification(notif.id)}
                              className={styles.closeNotif}
                            />
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {detectionNotifications.length > 0 && (
                      <div className={styles.notificationCategory}>
                        <h4>Person Detection</h4>
                        {detectionNotifications.map((notif) => (
                          <div key={notif.id} className={`${styles.notification} ${styles[notif.status || 'default']}`}>
                            {renderNotificationIcon(notif)}
                            <div className={styles.notificationContent}>
                              <p>{notif.message}</p>
                              <small>{new Date(notif.timestamp).toLocaleTimeString()}</small>
                            </div>
                            <MdClose 
                              onClick={() => removeNotification(notif.id)}
                              className={styles.closeNotif}
                            />
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {otherNotifications.length > 0 && (
                      <div className={styles.notificationCategory}>
                        <h4>Other</h4>
                        {otherNotifications.map((notif) => (
                          <div key={notif.id} className={`${styles.notification} ${styles.default}`}>
                            {renderNotificationIcon(notif)}
                            <div className={styles.notificationContent}>
                              <p>{notif.message}</p>
                              <small>{new Date(notif.timestamp).toLocaleTimeString()}</small>
                            </div>
                            <MdClose 
                              onClick={() => removeNotification(notif.id)}
                              className={styles.closeNotif}
                            />
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
          <div className={styles.themeToggle}>
            {isDarkMode ? (
              <MdOutlineLightMode 
                size={22} 
                onClick={handleThemeToggle} 
                title="Switch to light mode" 
                className={styles.icon}
              />
            ) : (
              <MdOutlineDarkMode 
                size={22} 
                onClick={handleThemeToggle} 
                title="Switch to dark mode" 
                className={styles.icon}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
