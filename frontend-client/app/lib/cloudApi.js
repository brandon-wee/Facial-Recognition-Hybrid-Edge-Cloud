/**
 * Cloud FastAPI Service
 * 
 * This module provides utility functions to interact with the cloud-based FastAPI service
 * running on the EC2 instance at 54.87.49.184:8000
 */

const CLOUD_API_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://192.168.146.198:8000";
console.log("[CLOUD API DEBUG] Using Cloud API URL:", CLOUD_API_URL);

// Increase timeouts to avoid premature aborts
const STATUS_TIMEOUT = 5000;  // Increased from 2000ms to 5000ms
const DEFAULT_TIMEOUT = 10000; // Increased from 5000ms to 10000ms

/**
 * Retry logic for API calls
 * @param {Function} apiFn - The function to call
 * @param {number} retries - Number of retries 
 * @param {number} delay - Delay between retries in ms
 */
const withRetry = async (apiFn, retries = 2, delay = 1000) => {
  let lastError;
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      if (attempt > 0) {
        console.log(`[CLOUD API DEBUG] Retry attempt ${attempt}/${retries}`);
      }
      return await apiFn();
    } catch (error) {
      console.error(`[CLOUD API ERROR] Attempt ${attempt + 1}/${retries + 1} failed:`, error);
      lastError = error;
      
      if (error.name === 'AbortError') {
        console.log('[CLOUD API DEBUG] Request was aborted due to timeout');
      }
      
      if (attempt < retries) {
        console.log(`[CLOUD API DEBUG] Waiting ${delay}ms before next retry...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError;
};

/**
 * Get real-time metadata from the edge device API
 */
export const getMetadata = async () => {
  return withRetry(async () => {
    console.log(`[CLOUD API DEBUG] Fetching metadata from: ${CLOUD_API_URL}/metadata`);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      console.log('[CLOUD API DEBUG] Aborting metadata request due to timeout');
    }, DEFAULT_TIMEOUT);
    
    try {
      const response = await fetch(`${CLOUD_API_URL}/metadata`, {
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache'
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      const data = await response.json();
      console.log("[CLOUD API DEBUG] Received metadata:", data);
      
      // Process metadata to extract names and best similarity score from the new format
      let names = [];
      let bestSim = 0;
      
      if (data.bboxes && Array.isArray(data.bboxes) && data.bboxes.length > 0) {
        // Extract names from bboxes objects (name is now a property)
        names = data.bboxes.map(bbox => bbox.name);
        
        // Find the highest similarity score (similarity is now a property)
        bestSim = Math.max(...data.bboxes.map(bbox => bbox.similarity || 0));
        bestSim = parseFloat(bestSim.toFixed(2)); // Format to 2 decimal places
      }
      
      return {
        fps: data.fps || 0,
        people_count: data.people_count || 0,
        names,
        best_sim: bestSim,
        // Include the original data for other components that might need it
        raw: data
      };
    } finally {
      clearTimeout(timeoutId);
    }
  });
};

/**
 * Get video feed URL from the edge device API
 */
export const getVideoFeedUrl = () => {
  return `${CLOUD_API_URL}/video_feed`;
};

/**
 * Process face embeddings for a user
 * @param {string} userId - The user ID
 * @param {File[]} images - Array of image files
 */
export const processUserEmbeddings = async (userId, images) => {
  return withRetry(async () => {
    // Get the user details first to get the username
    try {
      console.log(`[CLOUD API DEBUG] Processing embeddings for user ID: ${userId}`);
      
      // Create a FormData object to send the images
      const formData = new FormData();
      
      // The FastAPI endpoint expects "username" and "files" parameters
      formData.append('username', userId); // Using userId as username for now
      
      // Append each image to the form data with the correct field name "files"
      for (let i = 0; i < images.length; i++) {
        const image = images[i];
        
        // If the image is not PNG, try to convert it or add a flag
        if (image.type !== 'image/png') {
          console.log(`[CLOUD API DEBUG] Note: Image ${i} is not PNG format (${image.type})`);
        }
        
        formData.append('files', image); // Use "files" instead of "images"
      }
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        console.log('[CLOUD API DEBUG] Aborting embeddings processing request due to timeout');
      }, DEFAULT_TIMEOUT * 2); // Double timeout for image processing
      
      try {
        // Use the correct endpoint from the FastAPI server
        console.log(`[CLOUD API DEBUG] Sending request to: ${CLOUD_API_URL}/calculate_average_embedding`);
        const response = await fetch(`${CLOUD_API_URL}/calculate_average_embedding`, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });
        
        if (!response.ok) {
          const errorText = await response.text().catch(() => 'No error details available');
          throw new Error(`HTTP error ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        console.log("[CLOUD API DEBUG] Embeddings processing result:", result);
        
        return {
          status: result.status || 'success',
          message: result.message || 'Embeddings processed successfully',
          images: images.length,
          embedding: result.embedding
        };
      } finally {
        clearTimeout(timeoutId);
      }
    } catch (error) {
      console.error("[CLOUD API ERROR] Error processing embeddings:", error);
      throw error;
    }
  });
};

/**
 * Get cloud API status
 * Returns information about the API health and status
 */
export const getApiStatus = async () => {
  try {
    console.log("[CLOUD API DEBUG] Checking API status");
    
    return await withRetry(async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        console.log('[CLOUD API DEBUG] Aborting status check due to timeout');
      }, STATUS_TIMEOUT);
      
      try {
        const response = await fetch(`${CLOUD_API_URL}/status`, {
          headers: {
            'Accept': 'application/json',
          },
          signal: controller.signal
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        
        const status = await response.json();
        console.log("[CLOUD API DEBUG] API status:", status);
        return status;
      } finally {
        clearTimeout(timeoutId);
      }
    }, 1); // Only retry once for status
  } catch (error) {
    console.error("Error checking cloud API status:", error);
    return { 
      status: 'error', 
      message: error.name === 'AbortError' ? 'Connection timed out' : error.message 
    };
  }
};

/**
 * Get the base API URL for the cloud service
 * @returns {string} The base API URL
 */
export const getCloudApiUrl = () => {
  return CLOUD_API_URL;
};
