"use client";

import { useState, useEffect } from "react";
import styles from "@/app/ui/dashboard/settings/settings.module.css";
import { MdSave, MdRefresh, MdPersonAdd, MdSync, MdCamera } from "react-icons/md";
import Image from "next/image";

const SettingsPage = () => {
  // Form state
  const [formData, setFormData] = useState({
    email: "user@example.com",
    username: "admin",
    password: "",
    confirmPassword: "",
    theme: "system",
    emailNotifications: true,
    systemAlerts: true,
    updateNotifications: false,
    apiKey: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    webhookUrl: "",
    dataRetention: 30,
    modelAccuracyThreshold: 80,
    autoUpdate: true,
    privacyMode: "balanced"
  });
  
  // UI state
  const [activeTab, setActiveTab] = useState("account");
  const [isSaving, setIsSaving] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [passwordError, setPasswordError] = useState("");
  
  // Handle form input changes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value
    }));
    
    // Clear password error when user types in password fields
    if (name === "password" || name === "confirmPassword") {
      setPasswordError("");
    }
  };
  
  // Generate new API key
  const generateApiKey = () => {
    const newApiKey = 'xxxxxxxx-xxxx-xxxx-xxxx-'.replace(/x/g, () => 
      Math.floor(Math.random() * 16).toString(16)) + Date.now().toString(16);
    
    setFormData(prev => ({
      ...prev,
      apiKey: newApiKey
    }));
    
    // Show success notification briefly
    setShowSuccess(true);
    setTimeout(() => setShowSuccess(false), 3000);
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Password validation
    if (formData.password && formData.password !== formData.confirmPassword) {
      setPasswordError("Passwords do not match");
      return;
    }
    
    // Simulate API call
    setIsSaving(true);
    
    setTimeout(() => {
      setIsSaving(false);
      setShowSuccess(true);
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setShowSuccess(false);
      }, 3000);
    }, 1000);
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>Settings</h1>
        <button 
          className={`${styles.saveButton} ${isSaving ? styles.saving : ""}`} 
          onClick={handleSubmit}
          disabled={isSaving}
        >
          {isSaving ? (
            <>
              <MdSync className={styles.spinIcon} /> Saving...
            </>
          ) : (
            <>
              <MdSave /> Save Changes
            </>
          )}
        </button>
      </div>

      {showSuccess && (
        <div className={styles.successMessage}>
          Settings saved successfully!
        </div>
      )}

      <div className={styles.tabContainer}>
        <div className={styles.tabs}>
          <button 
            className={`${styles.tab} ${activeTab === "account" ? styles.activeTab : ""}`}
            onClick={() => setActiveTab("account")}
          >
            Account
          </button>
          <button 
            className={`${styles.tab} ${activeTab === "preferences" ? styles.activeTab : ""}`}
            onClick={() => setActiveTab("preferences")}
          >
            Preferences
          </button>
          <button 
            className={`${styles.tab} ${activeTab === "devices" ? styles.activeTab : ""}`}
            onClick={() => setActiveTab("devices")}
          >
            Edge Devices
          </button>
          <button 
            className={`${styles.tab} ${activeTab === "api" ? styles.activeTab : ""}`}
            onClick={() => setActiveTab("api")}
          >
            API Access
          </button>
        </div>
      </div>

      <div className={styles.sections}>
        {activeTab === "account" && (
          <div className={styles.section}>
            <h2>Account Settings</h2>
            
            <div className={styles.profileSection}>
              <div className={styles.profileImageContainer}>
                <Image 
                  src="/noavatar.png"
                  alt="Profile"
                  width={100}
                  height={100}
                  className={styles.profileImage}
                />
                <button className={styles.uploadButton}>
                  <MdCamera />
                </button>
              </div>
              <div className={styles.profileInfo}>
                <h3>{formData.username}</h3>
                <p>Administrator</p>
              </div>
            </div>
            
            <div className={styles.formGroup}>
              <label htmlFor="email">Email Address</label>
              <input 
                type="email" 
                id="email" 
                name="email" 
                value={formData.email} 
                onChange={handleChange}
              />
            </div>
            
            <div className={styles.formGroup}>
              <label htmlFor="username">Username</label>
              <input 
                type="text" 
                id="username" 
                name="username" 
                value={formData.username} 
                onChange={handleChange}
              />
            </div>
            
            <div className={styles.formRow}>
              <div className={styles.formGroup}>
                <label htmlFor="password">Change Password</label>
                <input 
                  type="password" 
                  id="password" 
                  name="password"
                  value={formData.password} 
                  onChange={handleChange} 
                  placeholder="Enter new password" 
                />
              </div>
              
              <div className={styles.formGroup}>
                <label htmlFor="confirmPassword">Confirm Password</label>
                <input 
                  type="password" 
                  id="confirmPassword" 
                  name="confirmPassword"
                  value={formData.confirmPassword} 
                  onChange={handleChange}
                  placeholder="Confirm new password" 
                />
              </div>
            </div>
            
            {passwordError && <div className={styles.errorMessage}>{passwordError}</div>}
            
            <div className={styles.dangerZone}>
              <h3>Danger Zone</h3>
              <button className={styles.dangerButton}>Delete Account</button>
            </div>
          </div>
        )}

        {activeTab === "preferences" && (
          <div className={styles.section}>
            <h2>Preferences</h2>
            
            <div className={styles.formGroup}>
              <label htmlFor="theme">Theme</label>
              <select 
                id="theme" 
                name="theme" 
                value={formData.theme} 
                onChange={handleChange}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="system">System Default</option>
              </select>
            </div>
            
            <div className={styles.formGroup}>
              <label>Notifications</label>
              <div className={styles.checkboxGroup}>
                <div className={styles.checkbox}>
                  <input 
                    type="checkbox" 
                    id="emailNotifications" 
                    name="emailNotifications"
                    checked={formData.emailNotifications}
                    onChange={handleChange}
                  />
                  <label htmlFor="emailNotifications">Email Notifications</label>
                </div>
                
                <div className={styles.checkbox}>
                  <input 
                    type="checkbox" 
                    id="systemAlerts" 
                    name="systemAlerts"
                    checked={formData.systemAlerts}
                    onChange={handleChange}
                  />
                  <label htmlFor="systemAlerts">System Alerts</label>
                </div>
                
                <div className={styles.checkbox}>
                  <input 
                    type="checkbox" 
                    id="updateNotifications" 
                    name="updateNotifications"
                    checked={formData.updateNotifications}
                    onChange={handleChange}
                  />
                  <label htmlFor="updateNotifications">Update Notifications</label>
                </div>
              </div>
            </div>
            
            <div className={styles.formGroup}>
              <label htmlFor="dataRetention">Data Retention (days)</label>
              <input 
                type="number" 
                id="dataRetention" 
                name="dataRetention"
                value={formData.dataRetention}
                onChange={handleChange}
                min="1" 
                max="365"
              />
            </div>
            
            <div className={styles.formGroup}>
              <label>Auto-Update System</label>
              <div className={styles.toggleSwitch}>
                <input 
                  type="checkbox" 
                  id="autoUpdate" 
                  name="autoUpdate"
                  checked={formData.autoUpdate}
                  onChange={handleChange}
                />
                <label htmlFor="autoUpdate"></label>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === "devices" && (
          <div className={styles.section}>
            <h2>Edge Device Settings</h2>
            
            <div className={styles.formGroup}>
              <label htmlFor="modelAccuracyThreshold">Model Accuracy Threshold (%)</label>
              <input 
                type="number" 
                id="modelAccuracyThreshold" 
                name="modelAccuracyThreshold"
                value={formData.modelAccuracyThreshold}
                onChange={handleChange}
                min="0" 
                max="100" 
              />
              <p className={styles.helperText}>
                Minimum accuracy threshold for deploying models to edge devices
              </p>
            </div>
            
            <div className={styles.formGroup}>
              <label htmlFor="privacyMode">Privacy Mode</label>
              <select 
                id="privacyMode" 
                name="privacyMode"
                value={formData.privacyMode}
                onChange={handleChange}
              >
                <option value="high">High (Process all data on edge device)</option>
                <option value="balanced">Balanced (Hybrid processing)</option>
                <option value="performance">Performance (Prioritize cloud processing)</option>
              </select>
            </div>
            
            <div className={styles.deviceManagementSection}>
              <div className={styles.sectionHeader}>
                <h3>Connected Devices</h3>
                <button className={styles.refreshButton}>
                  <MdRefresh /> Refresh
                </button>
              </div>
              
              <div className={styles.devicesList}>
                <div className={styles.deviceCard}>
                  <div className={styles.deviceIcon}>
                    <Image 
                      src="/arduino-icon.png" 
                      alt="Arduino Nicla" 
                      width={40} 
                      height={40} 
                    />
                  </div>
                  <div className={styles.deviceInfo}>
                    <h4>Arduino Nicla Sense</h4>
                    <p>Status: <span className={styles.active}>Online</span></p>
                    <p>Last sync: 5 minutes ago</p>
                  </div>
                  <button className={styles.secondaryButton}>Configure</button>
                </div>
                
                <div className={styles.deviceCard}>
                  <div className={styles.deviceIcon}>
                    <Image 
                      src="/raspberry-icon.png" 
                      alt="Raspberry Pi" 
                      width={40} 
                      height={40} 
                    />
                  </div>
                  <div className={styles.deviceInfo}>
                    <h4>Raspberry Pi 4B</h4>
                    <p>Status: <span className={styles.active}>Online</span></p>
                    <p>Last sync: 2 minutes ago</p>
                  </div>
                  <button className={styles.secondaryButton}>Configure</button>
                </div>
                
                <div className={styles.deviceCard}>
                  <div className={styles.deviceIcon}>
                    <Image 
                      src="/raspberry-icon.png" 
                      alt="Raspberry Pi" 
                      width={40} 
                      height={40} 
                    />
                  </div>
                  <div className={styles.deviceInfo}>
                    <h4>Raspberry Pi Zero</h4>
                    <p>Status: <span className={styles.inactive}>Offline</span></p>
                    <p>Last sync: 2 days ago</p>
                  </div>
                  <button className={styles.secondaryButton}>Configure</button>
                </div>
              </div>
              
              <button className={styles.addDeviceButton}>
                <MdPersonAdd /> Add New Device
              </button>
            </div>
          </div>
        )}
        
        {activeTab === "api" && (
          <div className={styles.section}>
            <h2>API Access</h2>
            
            <div className={styles.formGroup}>
              <label htmlFor="apiKey">API Key</label>
              <div className={styles.apiKeyGroup}>
                <input 
                  type="text" 
                  id="apiKey"
                  name="apiKey"
                  value={formData.apiKey} 
                  readOnly 
                />
                <button 
                  className={styles.generateButton}
                  onClick={generateApiKey}
                >
                  Generate New
                </button>
              </div>
              <p className={styles.warningText}>
                Generating a new API key will revoke access for any applications using the current key.
              </p>
            </div>
            
            <div className={styles.formGroup}>
              <label htmlFor="webhookUrl">Webhook URL</label>
              <input 
                type="url" 
                id="webhookUrl" 
                name="webhookUrl"
                value={formData.webhookUrl}
                onChange={handleChange}
                placeholder="https://your-webhook-endpoint.com" 
              />
              <p className={styles.helperText}>
                Receive real-time event notifications at this URL
              </p>
            </div>
            
            <div className={styles.apiDocs}>
              <h3>API Documentation</h3>
              <p>Access our comprehensive API documentation to integrate with our system.</p>
              <button className={styles.docsButton}>View Documentation</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;
