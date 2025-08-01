"use client";

import { updateUserEmbeddings } from "@/app/lib/actions";
import { processUserEmbeddings } from "@/app/lib/cloudApi";
import styles from "./embeddingUpdate.module.css";
import Image from "next/image";
import { useState, useRef, useEffect, useCallback } from "react";
import Webcam from "react-webcam"; // Add this import

const EmbeddingUpdate = ({ user }) => {
  // Safety check - create a safe user object
  const safeUser = {
    username: user?.username || "Unknown",
    email: user?.email || "",
    id: user?.id || user?._id || "",
    img: user?.img || "/noavatar.png",
    hasEmbeddings: !!(user?.embeddings?.length || user?.embeddings?.exists || user?.embeddingsUpdated)
  };
  
  const [images, setImages] = useState([]);
  const [cameraActive, setCameraActive] = useState(false);
  const [status, setStatus] = useState(null);
  const [showCapturePreview, setShowCapturePreview] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null); // New ref for Webcam component
  const isMounted = useRef(true);

  // Register when the component mounts and unmounts
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);
  
  // Define video constraints for the webcam
  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  // Start webcam
  const startCamera = useCallback(() => {
    if (!isMounted.current) return;
    
    console.log("[CAMERA] Attempting to start camera...");
    setCameraError(null);
    setStatus({ info: true, message: "Activating camera..." });
    setCameraActive(true);
    setShowCapturePreview(false);
    setCapturedImage(null);
  }, []);

  // Handle webcam errors
  const handleWebcamError = useCallback((err) => {
    if (!isMounted.current) return;
    
    console.error("[CAMERA] Webcam error:", err);
    setCameraActive(false);
    
    let errorMessage = "Camera access failed";
    
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      errorMessage = "Camera access denied. Please allow camera access in your browser settings.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      errorMessage = "No camera found on your device.";
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      errorMessage = "Camera is already in use by another application.";
    } else if (err.name === "OverconstrainedError") {
      errorMessage = "Camera cannot satisfy the requested constraints.";
    } else {
      errorMessage = `Camera error: ${err.message || "Unknown error"}`;
    }
    
    setCameraError(errorMessage);
    setStatus({ error: true, message: errorMessage });
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (webcamRef.current && webcamRef.current.stream) {
      const stream = webcamRef.current.stream;
      const tracks = stream.getTracks();
      
      tracks.forEach(track => track.stop());
      console.log("[CAMERA] Camera stopped");
    }
    
    setCameraActive(false);
  }, []);

  // Capture image from webcam
  const captureImage = useCallback(() => {
    if (!webcamRef.current) return;
    
    try {
      console.log("[CAMERA] Capturing image");
      
      // Capture image from webcam
      const imageSrc = webcamRef.current.getScreenshot();
      
      if (!imageSrc) {
        console.error("[CAMERA] Failed to capture screenshot");
        setStatus({ error: true, message: "Failed to capture image. Please try again." });
        return;
      }
      
      // Set preview
      setCapturedImage(imageSrc);
      setShowCapturePreview(true);
      
      // Convert base64 to file object
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], `capture_${Date.now()}.png`, { type: "image/png" });
          setImages(prev => [...prev, file]);
          console.log("[CAMERA] Image captured and added to collection");
        })
        .catch(err => {
          console.error("[CAMERA] Error processing captured image:", err);
          setStatus({ error: true, message: "Error processing image" });
        });
    } catch (err) {
      console.error("[CAMERA] Error capturing image:", err);
      setStatus({ error: true, message: `Failed to capture image: ${err.message}` });
    }
  }, [webcamRef]);

  // Handle successful webcam initialization
  const handleWebcamInit = useCallback(() => {
    if (isMounted.current) {
      console.log("[CAMERA] Webcam initialized successfully");
      setStatus({ success: true, message: "Camera activated successfully" });
    }
  }, []);

  // Confirm using captured image
  const confirmCapturedImage = useCallback(() => {
    setShowCapturePreview(false);
    stopCamera();
  }, [stopCamera]);

  // Retake the photo
  const retakePhoto = useCallback(() => {
    setShowCapturePreview(false);
    setCapturedImage(null);
    
    // Remove the last image if it was added
    if (images.length > 0) {
      setImages(prev => prev.slice(0, -1));
    }
  }, [images.length]);

  // Handle file upload
  const handleFileUpload = useCallback((e) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      setImages(prev => [...prev, ...filesArray]);
    }
  }, []);

  // Remove an image from the list
  const removeImage = useCallback((index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Submit the form with images to update embeddings using CloudAPI
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (images.length === 0) {
      setStatus({ error: true, message: "Please add at least one image" });
      return;
    }

    setStatus({ loading: true, message: "Calculating embeddings..." });
    
    console.log("Submitting with username:", safeUser.username);
    console.log("Number of images:", images.length);
    
    try {
      // Use the CloudAPI service with username instead of ID
      console.log("Sending request to CloudAPI for embedding processing");
      
      // Use the username since that's what the FastAPI endpoint expects
      const result = await processUserEmbeddings(safeUser.username, images);
      console.log("CloudAPI response:", result);
      
      if (result.status === "success") {
        console.log("CloudAPI success, updating local database with userId:", safeUser.id);
        
        // Update our local database to store the reference to the embedding
        const updateResult = await updateUserEmbeddings({
          userId: safeUser.id,
          embeddingUpdated: true
        });
        
        console.log("Local database update result:", updateResult);
        
        if (updateResult.success) {
          setStatus({ success: true, message: `Embeddings updated successfully! Used ${result.images} images.` });
          setImages([]);
        } else {
          setStatus({ warning: true, message: `Face data processed but database update failed: ${updateResult.message}` });
        }
      } else {
        setStatus({ error: true, message: result.message || "Face processing failed" });
      }
    } catch (error) {
      console.error("Error updating embeddings:", error);
      
      // Provide more specific error messages
      if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError") || error.name === "AbortError") {
        setStatus({ error: true, message: "Cannot connect to face recognition server. Is it running?" });
      } else if (error.message.includes("404")) {
        setStatus({ error: true, message: "API endpoint not found. Check server configuration." });
      } else {
        setStatus({ error: true, message: `Error: ${error.message}` });
      }
    } finally {
      setStatus(prev => ({ ...prev, loading: false }));
    }
  };

  return (
    <div className={styles.formContainer}>
      <div className={styles.header}>
        <h1>Face Recognition Setup</h1>
        <p className={styles.subtitle}>Upload or capture images to enable face recognition</p>
      </div>
      
      <div className={styles.userInfo}>
        <div className={styles.userImageContainer}>
          <Image
            src={safeUser.img}
            alt={safeUser.username}
            width={100}
            height={100}
            className={styles.userImage}
          />
        </div>
        <div className={styles.userDetails}>
          <h2>{safeUser.username}</h2>
          <p>{safeUser.email}</p>
          <p className={styles.embedStatus}>
            {safeUser.hasEmbeddings ? 
              <span className={styles.embeddingsAvailable}>✓ Embeddings available</span> : 
              <span className={styles.noEmbeddings}>✗ No embeddings available</span>}
          </p>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.inputSection}>
          <h3>Face Image Collection</h3>
          <p className={styles.embedInstructions}>
            Add multiple images of your face to improve recognition accuracy.
            Try to include different angles, expressions and lighting conditions.
          </p>
          
          <div className={styles.cardContainer}>
            <div className={styles.card}>
              <div className={styles.cardHeader}>
                <h4>Upload Files</h4>
                <span className={styles.cardIcon}>📁</span>
              </div>
              <div className={styles.cardContent}>
                <p>Select existing photos from your device</p>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileUpload}
                  ref={fileInputRef}
                  className={styles.fileInput}
                />
                <button 
                  type="button" 
                  onClick={() => fileInputRef.current?.click()}
                  className={styles.uploadButton}
                >
                  Browse Files
                </button>
              </div>
            </div>
            
            <div className={styles.card}>
              <div className={styles.cardHeader}>
                <h4>Use Camera</h4>
                <span className={styles.cardIcon}>📷</span>
              </div>
              <div className={styles.cardContent}>
                <p>Take photos using your device's camera</p>
                {!cameraActive && !showCapturePreview ? (
                  <button 
                    type="button" 
                    onClick={startCamera}
                    className={styles.cameraButton}
                  >
                    Start Camera
                  </button>
                ) : (
                  <div className={styles.enhancedCameraContainer}>
                    {showCapturePreview ? (
                      <div className={styles.capturePreview}>
                        <div className={styles.previewImageContainer}>
                          <img 
                            src={capturedImage} 
                            alt="Captured photo" 
                            className={styles.previewCapturedImage}
                          />
                        </div>
                        <div className={styles.capturePreviewControls}>
                          <button 
                            type="button" 
                            onClick={confirmCapturedImage}
                            className={`${styles.captureButton} ${styles.confirmButton}`}
                          >
                            Use Photo
                          </button>
                          <button 
                            type="button" 
                            onClick={retakePhoto}
                            className={`${styles.captureButton} ${styles.retakeButton}`}
                          >
                            Retake
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div 
                          className={styles.videoWrapper}
                          style={{ minHeight: "240px", display: "block" }}
                        >
                          {cameraError && (
                            <div className={styles.cameraErrorOverlay}>
                              <p>{cameraError}</p>
                              <button 
                                onClick={startCamera}
                                className={styles.retryButton}
                              >
                                Retry
                              </button>
                            </div>
                          )}
                          
                          <Webcam
                            audio={false}
                            ref={webcamRef}
                            videoConstraints={videoConstraints}
                            onUserMedia={handleWebcamInit}
                            onUserMediaError={handleWebcamError}
                            screenshotFormat="image/png"
                            className={styles.video}
                            mirrored={true}
                          />
                          
                          <div className={styles.cameraOverlay}>
                            <div className={styles.faceBoundary}></div>
                          </div>
                        </div>
                        <div className={styles.cameraControls}>
                          <button 
                            type="button" 
                            onClick={captureImage}
                            className={styles.captureButton}
                          >
                            Take Photo
                          </button>
                          <button 
                            type="button" 
                            onClick={stopCamera}
                            className={styles.stopButton}
                          >
                            Cancel
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
                
                {cameraActive && !showCapturePreview && !cameraError && (
                  <p className={styles.cameraStatus}>Camera is active. Position your face in the frame.</p>
                )}
              </div>
            </div>
          </div>
        </div>
        
        {images.length > 0 && (
          <div className={styles.imageGallerySection}>
            <h3>Selected Images <span className={styles.imageCount}>({images.length})</span></h3>
            <div className={styles.imageGallery}>
              {images.map((image, index) => (
                <div key={index} className={styles.galleryItem}>
                  <div className={styles.galleryImageContainer}>
                    <Image
                      src={URL.createObjectURL(image)}
                      alt={`Preview ${index}`}
                      width={120}
                      height={120}
                      className={styles.galleryImage}
                    />
                    <button 
                      type="button" 
                      onClick={() => removeImage(index)}
                      className={styles.removeButton}
                      title="Remove image"
                    >
                      ✕
                    </button>
                  </div>
                  <div className={styles.imageIndex}>Image {index + 1}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {status && (
          <div className={`${styles.statusMessage} ${
            status.error ? styles.error : 
            status.success ? styles.success : 
            status.loading ? styles.loading : 
            status.info ? styles.info : ""
          }`}>
            {status.message}
          </div>
        )}
        
        <div className={styles.formActions}>
          <button 
            type="submit" 
            disabled={images.length === 0 || status?.loading}
            className={styles.submitButton}
          >
            {status?.loading ? (
              <span className={styles.loadingButtonContent}>
                <span className={styles.spinner}></span>
                <span>Processing...</span>
              </span>
            ) : (
              <span>Update Face Embeddings</span>
            )}
          </button>
          {images.length > 0 && !status?.loading && (
            <p className={styles.readyMessage}>Ready to process {images.length} image(s)</p>
          )}
        </div>
      </form>
    </div>
  );
};

export default EmbeddingUpdate;
