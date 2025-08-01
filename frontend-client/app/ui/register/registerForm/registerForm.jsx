"use client";

import { registerUser } from "@/app/lib/actions";
import styles from "./registerForm.module.css";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const RegisterForm = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");
    
    // Validate passwords match
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      setIsLoading(false);
      return;
    }
    
    try {
      // Create FormData manually
      const formData = new FormData();
      formData.append("username", username);
      formData.append("email", email);
      formData.append("password", password);
      
      const result = await registerUser(formData);
      
      if (result.success) {
        // Redirect to login on success
        router.push("/login?registered=true");
      } else {
        setError(result.message || "Registration failed");
      }
    } catch (err) {
      console.error("Registration error:", err);
      setError("An error occurred during registration");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <h1>Create Account</h1>
      <div className={styles.inputContainer}>
        <input 
          type="text" 
          placeholder="Username" 
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required 
          aria-label="Username"
          className={error ? styles.errorInput : ""}
        />
      </div>
      <div className={styles.inputContainer}>
        <input 
          type="email" 
          placeholder="Email" 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required 
          aria-label="Email"
          className={error ? styles.errorInput : ""}
        />
      </div>
      <div className={styles.inputContainer}>
        <input 
          type="password" 
          placeholder="Password" 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          aria-label="Password"
          className={error ? styles.errorInput : ""}
        />
      </div>
      <div className={styles.inputContainer}>
        <input 
          type="password" 
          placeholder="Confirm Password" 
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          aria-label="Confirm Password"
          className={error ? styles.errorInput : ""}
        />
      </div>
      <button 
        type="submit" 
        disabled={isLoading || !username || !email || !password || !confirmPassword}
        className={isLoading ? styles.loadingButton : ""}
      >
        {isLoading ? "Registering..." : "Register"}
      </button>
      {error && <div className={styles.error}>{error}</div>}
      <div className={styles.loginPrompt}>
        Already have an account? <Link href="/login">Login</Link>
      </div>
    </form>
  );
};

export default RegisterForm;
