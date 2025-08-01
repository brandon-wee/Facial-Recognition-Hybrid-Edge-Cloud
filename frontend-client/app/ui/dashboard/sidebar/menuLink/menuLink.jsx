"use client";

import Link from "next/link";
import styles from "./menuLink.module.css";
import { usePathname } from "next/navigation";
import { MdLogout } from "react-icons/md";
import { useRouter } from "next/navigation";

const MenuLink = ({ item }) => {
  const pathname = usePathname();
  const router = useRouter();
  
  const isActive = pathname === item.path;

  // Handle logout client-side if this is a logout link
  const handleClick = async (e) => {
    if (item.title === "Logout") {
      e.preventDefault();
      
      // Clear the auth cookie client-side
      document.cookie = "auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
      
      // Redirect to login
      router.push("/login");
      router.refresh();
    }
  };

  return (
    <Link href={item.path} className={`${styles.container} ${isActive && styles.active}`} onClick={handleClick}>
      {item.icon}
      {item.title}
    </Link>
  );
};

export default MenuLink;