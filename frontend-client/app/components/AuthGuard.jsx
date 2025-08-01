"use client";

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';

export default function AuthGuard({ children }) {
  const router = useRouter();
  const pathname = usePathname();
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    // Check if user is authenticated
    const checkAuth = () => {
      console.log("[AUTH GUARD] Checking authentication status...");
      
      // First check localStorage
      const username = localStorage.getItem('username');
      console.log("[AUTH GUARD] Username in localStorage:", username);
      
      // Then check cookies as backup
      const cookies = document.cookie;
      const hasAuthCookie = cookies.includes('auth-token=');
      const hasUserInfoCookie = cookies.includes('user-info=');
      
      console.log("[AUTH GUARD] Auth cookies present:", { hasAuthCookie, hasUserInfoCookie });
      
      // Determine if logged in based on both sources
      const isLoggedIn = 
        (username && username !== 'Guest' && username !== 'null') || 
        hasAuthCookie || 
        hasUserInfoCookie;
      
      console.log("[AUTH GUARD] Authentication status:", isLoggedIn ? "Logged in" : "Not logged in");
      
      // If on login page and already logged in, redirect to dashboard
      if ((pathname === '/login' || pathname === '/register' || pathname === '/reset-password') && isLoggedIn) {
        console.log("[AUTH GUARD] Already logged in, redirecting from login page to dashboard");
        router.replace('/dashboard');
        return;
      }
      
      // If on dashboard pages and not logged in, redirect to login
      if (pathname.includes('/dashboard') && !isLoggedIn) {
        console.log("[AUTH GUARD] Not logged in, redirecting from dashboard to login");
        router.replace('/login');
        return;
      }
      
      setIsLoading(false);
    };
    
    checkAuth();
    
    // Listen for storage events (like localStorage changes)
    const handleStorageChange = () => {
      console.log("[AUTH GUARD] Storage changed, rechecking auth status");
      checkAuth();
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [pathname, router]);
  
  // Show loading indicator while checking authentication
  if (isLoading) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading...</div>;
  }
  
  return children;
}
