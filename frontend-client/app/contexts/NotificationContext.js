"use client";
import { createContext, useContext, useState, useCallback } from "react";

// Create the context
const NotificationContext = createContext();

// Provider component
export function NotificationProvider({ children }) {
  const [notifications, setNotifications] = useState([]);

  // Add a new notification
  const addNotification = useCallback((notification) => {
    setNotifications(prev => [...prev, notification]);
  }, []);

  // Remove a notification by id
  const removeNotification = useCallback((id) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  // Value to be provided to consumers
  const value = {
    notifications,
    addNotification,
    removeNotification
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
}

// Custom hook for consuming the context
export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error("useNotifications must be used within a NotificationProvider");
  }
  return context;
}
