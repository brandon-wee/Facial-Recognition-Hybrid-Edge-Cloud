"use client";
import { createContext, useState, useContext, useEffect } from 'react';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  // Use state to track dark mode status
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Initialize theme from localStorage on component mount (client-side only)
  useEffect(() => {
    // Check if we're in the browser environment
    if (typeof window !== 'undefined') {
      const savedTheme = localStorage.getItem('darkMode');
      
      // Use saved preference if available, otherwise use system preference
      if (savedTheme !== null) {
        setIsDarkMode(savedTheme === 'true');
      } else {
        // Check system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setIsDarkMode(prefersDark);
      }
    }
  }, []);

  // Update document class and localStorage when theme changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Apply theme class to document root
      if (isDarkMode) {
        document.documentElement.classList.remove('light');
      } else {
        document.documentElement.classList.add('light');
      }
      
      // Save preference to localStorage
      localStorage.setItem('darkMode', isDarkMode);
    }
  }, [isDarkMode]);

  // Toggle theme function
  const toggleTheme = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  return (
    <ThemeContext.Provider value={{ isDarkMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook to use theme
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};