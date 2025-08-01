"use client";
import { createContext, useContext, useState } from 'react';

const NotificationContext = createContext();

export function NotificationProvider({ children }) {
  const [notifications, setNotifications] = useState([]);

  const addNotification = (type, message) => {
    const newNotification = {
      id: Date.now(),
      type,
      message,
      timestamp: new Date(),
    };
    setNotifications(prev => [newNotification, ...prev]);
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  };

  const clearNotifications = () => {
    setNotifications([]);
  };

  return (
    <NotificationContext.Provider value={{ 
      notifications, 
      addNotification, 
      removeNotification,
      clearNotifications 
    }}>
      {children}
    </NotificationContext.Provider>
  );
}

export const useNotifications = () => useContext(NotificationContext);