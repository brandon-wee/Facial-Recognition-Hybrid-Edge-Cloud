import mongoose from "mongoose";

export const connectToDB = async () => {
  try {
    if (mongoose.connection.readyState === 1) {
      return mongoose.connection;
    }
    
    const MONGO_URL = process.env.MONGO || "mongodb://127.0.0.1:27017/nextadmin";
    
    if (!MONGO_URL) {
      throw new Error("Please define the MONGO environment variable");
    }
    
    const conn = await mongoose.connect(MONGO_URL, {});
    console.log("MongoDB connected successfully");
    return conn;
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
    throw error;
  }
};