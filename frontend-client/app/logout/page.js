"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function LogoutPage() {
  const router = useRouter();
  
  useEffect(() => {
    // Delete the auth cookie (client-side)
    document.cookie = "auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    
    // Small delay before redirect to ensure cookie is cleared
    setTimeout(() => {
      router.push('/login');
    }, 100);
  }, [router]);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{textAlign: 'center'}}>
        <h1>Logging out...</h1>
        <p>Redirecting to login page</p>
      </div>
    </div>
  );
}
