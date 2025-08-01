"use client";

import { useEffect, useRef, useState } from 'react';
import Script from 'next/script';
import styles from "../ui/login/login.module.css";
import ResetPasswordForm from "../ui/reset-password/resetPasswordForm/resetPasswordForm";

const ResetPasswordPage = () => {
  const vantaRef = useRef(null);
  const vantaEffectRef = useRef(null);
  const [scriptsLoaded, setScriptsLoaded] = useState({
    three: false,
    vanta: false
  });
  const [backgroundInitialized, setBackgroundInitialized] = useState(false);

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

  return (
    <>
      <Script 
        src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
        strategy="afterInteractive"
        onLoad={() => setScriptsLoaded(prev => ({ ...prev, three: true }))}
        onError={() => console.error("Failed to load Three.js")}
      />
      
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
        <ResetPasswordForm />
      </div>
    </>
  );
};

export default ResetPasswordPage;
