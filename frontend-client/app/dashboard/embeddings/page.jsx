import { findUserByUsername } from "@/app/lib/actions";
import EmbeddingUpdate from "@/app/ui/dashboard/embeddings/embeddingUpdate";
import styles from "@/app/ui/dashboard/embeddings/embeddingUpdate.module.css";
import { Suspense } from "react";

// SearchParams component to handle user search and display the embedding update form
const UserEmbeddingsPage = async ({ searchParams }) => {
  const username = searchParams?.username || "";
  let user = null;
  
  if (username) {
    try {
      user = await findUserByUsername(username);
    } catch (err) {
      console.error("Error finding user:", err);
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.top}>
        <h1 className={styles.title}>Update User Embeddings</h1>
        <p className={styles.desc}>
          Search for a user and update their face embeddings by uploading images or using the camera.
        </p>
      </div>
      
      <div className={styles.search}>
        <form className={styles.searchForm}>
          <input 
            type="text" 
            name="username" 
            placeholder="Search by username..."
            defaultValue={username}
            className={styles.searchInput}
          />
          <button type="submit" className={styles.searchButton}>Search</button>
        </form>
      </div>
      
      {username && !user && (
        <div className={styles.notFound}>User not found. Please try a different username.</div>
      )}
      
      {user && (
        <Suspense fallback={<div className={styles.loading}>Loading embedding update form...</div>}>
          <EmbeddingUpdate user={user} />
        </Suspense>
      )}
    </div>
  );
};

export default UserEmbeddingsPage;
