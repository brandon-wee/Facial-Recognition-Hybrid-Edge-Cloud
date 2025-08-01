"use client";

import { useEffect, useRef, useState } from 'react';
import Script from 'next/script';
import styles from "@/app/ui/login/login.module.css";
import LoginForm from "../ui/login/loginForm/loginForm";
import { useRouter } from "next/navigation";

const LoginPage = () => {
  const vantaRef = useRef(null);
  const vantaEffectRef = useRef(null);
  const [scriptsLoaded, setScriptsLoaded] = useState({
    three: false,
    vanta: false
  });
  const [backgroundInitialized, setBackgroundInitialized] = useState(false);
  const [debugInfo, setDebugInfo] = useState('');
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  
  // Function to initialize VANTA effect
  const initVantaEffect = () => {
    if (!vantaRef.current || vantaEffectRef.current) return;
    
    try {
      // Check if VANTA is available in the window object
      if (window.VANTA && window.VANTA.NET) {
        vantaEffectRef.current = window.VANTA.NET({
          el: vantaRef.current,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0x3a3a5c,
          backgroundColor: 0x10101a,
          maxDistance: 22.00,
          spacing: 19.00
        });
        setBackgroundInitialized(true);
      }
    } catch (error) {
      console.error("Error initializing VANTA effect:", error);
    }
  };

  // Initialize VANTA effect once both scripts are loaded
  useEffect(() => {
    // If scripts are already loaded in the browser (from cache), initialize immediately
    if (window.THREE && window.VANTA && !vantaEffectRef.current) {
      setScriptsLoaded({ three: true, vanta: true });
      initVantaEffect();
    }
    
    // Add a debug utility function
    window.showLoginDebug = () => {
      const cookies = document.cookie.split(';').map(c => c.trim());
      setDebugInfo(`Cookies: ${cookies.join(', ')}`);
    };

    // Cleanup function that runs when component unmounts
    return () => {
      if (vantaEffectRef.current) {
        vantaEffectRef.current.destroy();
        vantaEffectRef.current = null;
        setBackgroundInitialized(false);
      }
    };
  }, []);

  // Effect to initialize VANTA when scripts load
  useEffect(() => {
    if (scriptsLoaded.three && scriptsLoaded.vanta) {
      initVantaEffect();
    }
  }, [scriptsLoaded]);

  // Check if user is already logged in
  useEffect(() => {
    console.log("[LOGIN PAGE] Checking if user is already logged in");
    
    const checkAuthentication = () => {
      // Check localStorage
      const username = localStorage.getItem('username');
      
      // Check cookies
      const hasAuthCookie = document.cookie.includes('auth-token=');
      
      // Determine auth status
      const isLoggedIn = 
        (username && username !== 'Guest' && username !== 'null') || 
        hasAuthCookie;
      
      console.log("[LOGIN PAGE] Auth check:", { isLoggedIn, username, hasAuthCookie });
      
      if (isLoggedIn) {
        console.log("[LOGIN PAGE] Already logged in, redirecting to dashboard");
        router.replace('/dashboard');
      } else {
        setIsLoading(false);
      }
    };
    
    checkAuthentication();
  }, [router]);
  
  if (isLoading) {
    return <div className={styles.container}><p>Loading...</p></div>;
  }
  
  return (
    <>
      {/* Load Three.js from CDN with afterInteractive strategy */}
      <Script 
        src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
        strategy="afterInteractive"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, three: true }))}
        onError={() => console.error("Failed to load Three.js")}
      />
      
      {/* Load VANTA.js from CDN with afterInteractive strategy */}
      <Script 
        src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"
        strategy="afterInteractive"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, vanta: true }))}
        onError={() => console.error("Failed to load VANTA.js")}
      />
      
      <div 
        className={`${styles.container} ${!backgroundInitialized ? styles.fallbackBackground : ''}`} 
        ref={vantaRef}
      >
        <LoginForm />
        
        {/* Debug button (hidden in production) */}
        {process.env.NODE_ENV !== 'production' && (
          <div style={{ position: 'absolute', bottom: '10px', right: '10px', zIndex: 1000 }}>
            <button onClick={() => window.showLoginDebug()}>Debug</button>
            {debugInfo && <pre style={{ background: '#fff', padding: '10px' }}>{debugInfo}</pre>}
          </div>
        )}
      </div>
    </>
  );
};

export default LoginPage;