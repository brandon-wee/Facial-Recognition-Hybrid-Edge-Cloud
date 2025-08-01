import { cookies } from 'next/headers';
import { User } from "./models";
import { connectToDB } from "./utils";

// Simple token generation
const generateToken = (payload) => {
  const data = JSON.stringify(payload);
  const buffer = Buffer.from(data).toString('base64');
  return buffer;
};

// Simple token verification
const verifyToken = (token) => {
  try {
    const data = Buffer.from(token, 'base64').toString();
    return JSON.parse(data);
  } catch (error) {
    return null;
  }
};

// Debug function to log without exposing passwords
const debugLog = (message, data) => {
  console.log(`[AUTH DEBUG] ${message}`, data);
};

// Login function to authenticate user
export async function loginUser(credentials) {
  debugLog("Login attempt with username:", credentials.username);
  
  try {
    await connectToDB();
    
    // Find user by username and explicitly select password field
    const user = await User.findOne({ username: credentials.username }).select('+password');
    
    if (!user) {
      debugLog("User not found");
      return { success: false, message: "User not found" };
    }
    
    debugLog("User found:", { 
      id: user._id, 
      username: user.username,
      hasPassword: !!user.password
    });
    
    // Debug the password comparison
    console.log("[AUTH DEBUG] Stored password:", user.password);
    console.log("[AUTH DEBUG] Provided password:", credentials.password);
    
    // Simple string comparison - no hashing
    const passwordMatches = credentials.password === user.password;
    debugLog("Password match:", passwordMatches);
    
    if (!passwordMatches) {
      return { success: false, message: "Invalid password" };
    }
    
    // Create user data object for the token
    const userData = {
      id: user._id.toString(),
      username: user.username,
      isAdmin: !!user.isAdmin
    };
    
    console.log("[AUTH DEBUG] User data with username before token creation:", userData);
    
    // Convert to JSON string
    const token = JSON.stringify(userData);
    console.log("[AUTH DEBUG] FINAL TOKEN:", token);
    debugLog("Login successful, token created with username:", userData.username);
    
    return { 
      success: true, 
      token: token,
      user: userData
    };
  } catch (err) {
    console.error("[AUTH ERROR]", err);
    return { success: false, message: "Authentication error: " + err.message };
  }
}

// Get the current user from the cookie
export function getUser() {
  const cookieStore = cookies();
  const token = cookieStore.get('auth-token')?.value;
  
  if (!token) return null;
  
  const userData = verifyToken(token);
  
  // Check if token has expired
  if (userData && userData.exp && userData.exp < Date.now()) {
    return null; // Token has expired
  }
  
  return userData;
}

// Check if the user is authenticated
export function isAuthenticated() {
  const user = getUser();
  return !!user;
}

// Check if the user is an admin
export function isAdmin() {
  const user = getUser();
  return user?.isAdmin === true;
}

// Logout function
export function logout() {
  cookies().delete('auth-token');
}

// Function to get user by username for direct user info retrieval
export const getUserByUsername = async (username) => {
  try {
    await connectToDB();
    const user = await User.findOne({ username }).lean();
    
    if (!user) {
      return null;
    }
    
    // Convert MongoDB ObjectId to string
    const userData = {
      ...user,
      _id: user._id.toString()
    };
    
    return userData;
  } catch (error) {
    console.error("[AUTH ERROR] Error fetching user by username:", error);
    throw error;
  }
};
