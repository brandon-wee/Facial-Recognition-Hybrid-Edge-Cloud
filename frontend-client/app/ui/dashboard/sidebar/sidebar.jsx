"use client";

import Image from "next/image";
import MenuLink from "./menuLink/menuLink";
import styles from "./sidebar.module.css";
import {
  MdDashboard,
  MdSupervisedUserCircle,
  MdWork,
  MdAnalytics,
  MdPeople,
  MdOutlineSettings,
  MdHelpCenter,
  MdLogout,
  MdOutlineCategory,
  MdFace,
  MdCircle,
  MdAdminPanelSettings,
} from "react-icons/md";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";

const menuItems = [
  {
    title: "Pages",
    list: [
      {
        title: "Dashboard",
        path: "/dashboard",
        icon: <MdDashboard />,
      },
      {
        title: "Users",
        path: "/dashboard/users",
        icon: <MdSupervisedUserCircle />,
      },
      {
        title: "Models",
        path: "/dashboard/models",
        icon: <MdOutlineCategory />,
      },
    ],
  },
  {
    title: "Analytics",
    list: [
      {
        title: "Real-time Data",
        path: "/dashboard/analytics/realtime",
        icon: <MdAnalytics />,
      },
      {
        title: "Edge Devices",
        path: "/dashboard/analytics/edge",
        icon: <MdWork />,
      },
    ],
  },
  {
    title: "People",
    list: [
      {
        title: "People",
        path: "/dashboard/people",
        icon: <MdPeople />,
      },
      {
        title: "Embeddings",
        path: "/dashboard/embeddings",
        icon: <MdFace />,
      },
    ],
  },
  {
    title: "User",
    list: [
      {
        title: "Settings",
        path: "/dashboard/settings",
        icon: <MdOutlineSettings />,
      },
      {
        title: "Help",
        path: "/dashboard/help",
        icon: <MdHelpCenter />,
      },
    ],
  },
];

const Sidebar = () => {
  const router = useRouter();
  const [user, setUser] = useState({
    username: "Loading...",
    img: null,
    isAdmin: false
  });
  
  // Improved user data loading with better debugging
  useEffect(() => {
    console.log("[SIDEBAR] Initializing user data...");
    
    // Function to get user data from available sources
    const loadUserData = () => {
      // Try localStorage first (most reliable after login)
      try {
        const storedUsername = localStorage.getItem('username');
        console.log("[SIDEBAR] localStorage username check:", storedUsername);
        
        if (storedUsername && storedUsername !== "Guest" && storedUsername !== "null") {
          const isAdmin = localStorage.getItem('isAdmin') === 'true';
          console.log("[SIDEBAR] Found valid username in localStorage:", storedUsername);
          
          setUser({
            username: storedUsername,
            img: null,
            isAdmin: isAdmin
          });
          return true;
        }
      } catch (e) {
        console.error("[SIDEBAR] Error reading from localStorage:", e);
      }
      
      // If not in localStorage, try cookies
      try {
        console.log("[SIDEBAR] Checking cookies for user data...");
        const cookies = document.cookie.split(';');
        const userInfoCookie = cookies.find(c => c.trim().startsWith('user-info='));
        
        if (userInfoCookie) {
          const cookieValue = userInfoCookie.split('=')[1];
          console.log("[SIDEBAR] Found user-info cookie:", cookieValue);
          
          try {
            const userData = JSON.parse(decodeURIComponent(cookieValue));
            if (userData && userData.username) {
              console.log("[SIDEBAR] Using username from cookie:", userData.username);
              
              // Update localStorage from cookie for consistency
              localStorage.setItem('username', userData.username);
              localStorage.setItem('isAdmin', userData.isAdmin ? 'true' : 'false');
              
              setUser({
                username: userData.username,
                img: null,
                isAdmin: !!userData.isAdmin
              });
              return true;
            }
          } catch (parseError) {
            console.error("[SIDEBAR] Error parsing cookie:", parseError);
          }
        } else {
          console.log("[SIDEBAR] No user-info cookie found");
        }
      } catch (cookieError) {
        console.error("[SIDEBAR] Error checking cookies:", cookieError);
      }
      
      // If we reach here, no valid user data found
      console.log("[SIDEBAR] No valid user data found, user might be logged out");
      setUser({
        username: "Guest",
        img: null,
        isAdmin: false
      });
      
      // Check if we should redirect to login
      if (window.location.pathname.includes('/dashboard')) {
        console.log("[SIDEBAR] No auth data found, redirecting to login");
        router.push('/login');
      }
      
      return false;
    };
    
    // Load user data immediately and set up refresh interval
    loadUserData();
    
    // Refresh user data every few seconds to catch any changes
    const refreshInterval = setInterval(() => {
      const username = localStorage.getItem('username');
      if (username && username !== user.username) {
        console.log("[SIDEBAR] Username changed in localStorage, updating:", username);
        loadUserData();
      }
    }, 2000);
    
    return () => clearInterval(refreshInterval);
  }, [router]);
  
  // Enhanced logout handler with better debugging
  const handleLogout = async () => {
    console.log("[SIDEBAR] Logging out user:", user.username);
    
    // Clear all authentication data
    
    // 1. Clear localStorage items
    console.log("[SIDEBAR] Clearing localStorage...");
    localStorage.removeItem('username');
    localStorage.removeItem('isAdmin');
    localStorage.clear(); // Clear everything to be thorough
    
    // 2. Clear cookies with different path configurations
    console.log("[SIDEBAR] Clearing cookies...");
    document.cookie = "auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    document.cookie = "user-info=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    document.cookie = "auth-token=; path=/dashboard; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    document.cookie = "user-info=; path=/dashboard; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    document.cookie = "auth-token=; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    document.cookie = "user-info=; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    
    // 3. Clear session storage
    console.log("[SIDEBAR] Clearing sessionStorage...");
    sessionStorage.clear();
    
    // 4. Update local state to reflect logout
    console.log("[SIDEBAR] Updating user state to Guest");
    setUser({
      username: "Guest",
      img: null,
      isAdmin: false
    });
    
    // Verify logout was successful
    setTimeout(() => {
      console.log("[SIDEBAR] Logout verification:");
      console.log("- localStorage username:", localStorage.getItem('username'));
      console.log("- Cookies:", document.cookie);
      
      if (!localStorage.getItem('username') && !document.cookie.includes('user-info')) {
        console.log("[SIDEBAR] Logout successful, no user data found");
      } else {
        console.warn("[SIDEBAR] Logout may be incomplete, some data remains");
      }
    }, 100);
    
    // 5. Redirect to login page
    console.log("[SIDEBAR] Redirecting to login page");
    router.push("/login");
  };
  
  return (
    <div className={styles.container}>
      <div className={styles.profileContainer}>
        <div className={styles.avatarWrapper}>
          <Image
            className={styles.avatar}
            src={user?.img || "/noavatar.png"}
            alt="Profile"
            width="50"
            height="50"
            priority
          />
          <div className={styles.statusDot}></div>
        </div>
        <div className={styles.profileInfo}>
          <h3 className={styles.profileName}>{user?.username || "Guest"}</h3>
          {user?.isAdmin ? (
            <div className={styles.roleBadge}>
              <MdAdminPanelSettings />
              <span>Admin</span>
            </div>
          ) : (
            <div className={styles.roleBadge}>
              <span>Member</span>
            </div>
          )}
        </div>
      </div>
      <ul className={styles.list}>
        {menuItems.map((cat) => (
          <li key={cat.title}>
            <span className={styles.cat}>{cat.title}</span>
            {cat.list.map((item) => (
              <MenuLink item={item} key={item.title} />
            ))}
          </li>
        ))}
      </ul>
      <button onClick={handleLogout} className={styles.logout}>
        <MdLogout />
        Logout
      </button>
    </div>
  );
};

export default Sidebar;