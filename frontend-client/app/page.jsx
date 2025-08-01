import { redirect } from "next/navigation";
import { getUser } from "./lib/auth"; // Import our custom auth function
import { cookies } from "next/headers";

export default async function Homepage() {
  // Use our custom getUser function instead of auth()
  const user = getUser();
  
  // Redirect to dashboard if logged in, otherwise to login page
  if (user) {
    redirect("/dashboard");
  } else {
    redirect("/login");
  }
  
  // This won't actually render, but included just in case
  return (
    <div>Redirecting...</div>
  );
}