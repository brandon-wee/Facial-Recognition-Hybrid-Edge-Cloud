"use client";

import { useEffect } from 'react';
import DashboardLayout from './layout';
import AuthGuard from '../components/AuthGuard';
import { useRouter } from 'next/navigation';

export default function ProtectedDashboardLayout({ children }) {
  const router = useRouter();
  
  // Additional check for authentication at this level
  useEffect(() => {
    console.log("[LAYOUT WRAPPER] Checking authentication status");
    
    // Function to check if user is authenticated
    const checkAuthenticated = () => {
      const username = localStorage.getItem('username');
      const hasAuthCookie = document.cookie.includes('auth-token=');
      
      const isAuthenticated = 
        (username && username !== 'Guest' && username !== 'null') || 
        hasAuthCookie;
      
      console.log("[LAYOUT WRAPPER] Auth status:", { 
        isAuthenticated, 
        username, 
        hasAuthCookie 
      });
      
      if (!isAuthenticated) {
        console.log("[LAYOUT WRAPPER] Not authenticated, redirecting to login");
        router.replace('/login');
      }
    };
    
    // Check auth status immediately
    checkAuthenticated();
    
    // Set up interval to periodically check auth status
    const intervalId = setInterval(checkAuthenticated, 5000);
    
    return () => clearInterval(intervalId);
  }, [router]);
  
  return (
    <AuthGuard>
      <DashboardLayout>{children}</DashboardLayout>
    </AuthGuard>
  );
}
