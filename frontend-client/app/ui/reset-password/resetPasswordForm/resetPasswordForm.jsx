"use client";

import { resetPassword } from "@/app/lib/actions";
import styles from "./resetPasswordForm.module.css";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const ResetPasswordForm = () => {
  const [email, setEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");
    
    // Validate passwords match
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match");
      setIsLoading(false);
      return;
    }
    
    try {
      // Create FormData manually
      const formData = new FormData();
      formData.append("email", email);
      formData.append("newPassword", newPassword);
      
      const result = await resetPassword(formData);
      
      if (result.success) {
        // Redirect to login on success
        router.push("/login?reset=true");
      } else {
        setError(result.message || "Password reset failed");
      }
    } catch (err) {
      console.error("Password reset error:", err);
      setError("An error occurred during password reset");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={styles.form}>
      <h1>Reset Password</h1>
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
          placeholder="New Password" 
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          required
          aria-label="New Password"
          className={error ? styles.errorInput : ""}
        />
      </div>
      <div className={styles.inputContainer}>
        <input 
          type="password" 
          placeholder="Confirm New Password" 
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          aria-label="Confirm New Password"
          className={error ? styles.errorInput : ""}
        />
      </div>
      <button 
        type="submit" 
        disabled={isLoading || !email || !newPassword || !confirmPassword}
        className={isLoading ? styles.loadingButton : ""}
      >
        {isLoading ? "Resetting..." : "Reset Password"}
      </button>
      {error && <div className={styles.error}>{error}</div>}
      <div className={styles.loginPrompt}>
        Remember your password? <Link href="/login">Login</Link>
      </div>
    </form>
  );
};

export default ResetPasswordForm;
