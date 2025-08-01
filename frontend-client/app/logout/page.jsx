"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function LogoutPage() {
  const router = useRouter();
  
  useEffect(() => {
    // Delete the auth cookie client-side
    document.cookie = "auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    
    // Redirect to login page
    setTimeout(() => {
      router.push('/login');
      router.refresh();
    }, 100);
  }, [router]);

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh',
      flexDirection: 'column',
      gap: '20px' 
    }}>
      <h1>Logging out...</h1>
      <p>Redirecting to login page...</p>
    </div>
  );
}
