import { updateUser } from "@/app/lib/actions";
import { fetchUser } from "@/app/lib/data";
import styles from "@/app/ui/dashboard/users/singleUser/singleUser.module.css";
import Image from "next/image";
import { connectToDB } from "@/app/lib/utils";
import { User } from "@/app/lib/models";
import EmbeddingUpdate from "@/app/ui/dashboard/embeddings/embeddingUpdate";

// Helper function to serialize MongoDB document
function serializeUser(user) {
  if (!user) return null;
  
  const serialized = {};
  
  // Convert MongoDB document to plain object if needed
  const userObj = typeof user.toObject === 'function' ? user.toObject() : user;
  
  // Copy all properties
  Object.keys(userObj).forEach(key => {
    serialized[key] = userObj[key];
  });
  
  // Ensure ID is properly serialized
  if (userObj._id) {
    serialized.id = userObj._id.toString();
  }
  
  // Convert dates to strings
  if (userObj.createdAt instanceof Date) {
    serialized.createdAt = userObj.createdAt.toISOString();
  }
  if (userObj.updatedAt instanceof Date) {
    serialized.updatedAt = userObj.updatedAt.toISOString();
  }
  
  console.log("User serialized:", serialized.id, serialized.username);
  
  return serialized;
}

// This is a server component
const SingleUserPage = async ({ params }) => {
  
  const { id } = params;
  const user = await fetchUser(id);
  
  await connectToDB();
  
  // Fetch user data
  const userData = await User.findById(id);
  
  // Serialize the user data to remove Mongoose-specific properties
  const serializedUser = serializeUser(userData);

  return (
    <div className={styles.container}>
      <div className={styles.infoContainer}>
        <h2 className={styles.title}>User Profile</h2>
        <div className={styles.imgContainer}>
          <Image src={user.img || "/noavatar.png"} alt={`${user.username}'s profile`} fill className={styles.userImage} />
        </div>
        <div className={styles.userInfo}>
          <h3>{user.username}</h3>
          <p className={styles.userRole}>{user.isAdmin ? "Administrator" : "Regular User"}</p>
          <p className={styles.userStatus}>Status: <span className={user.isActive ? styles.active : styles.inactive}>{user.isActive ? "Active" : "Inactive"}</span></p>
        </div>
      </div>
      
      <div className={styles.formContainer}>
        <h2 className={styles.title}>Edit User Information</h2>
        <form action={updateUser} className={styles.form}>
          <input type="hidden" name="id" value={user.id}/>
          
          <div className={styles.formGroup}>
            <label htmlFor="username">Username</label>
            <input type="text" id="username" name="username" placeholder={user.username} />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="email">Email</label>
            <input type="email" id="email" name="email" placeholder={user.email} />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="password">Password</label>
            <input type="password" id="password" name="password" placeholder="••••••••" />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="phone">Phone</label>
            <input type="text" id="phone" name="phone" placeholder={user.phone || "Not provided"} />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="isAdmin">Admin Status</label>
            <select name="isAdmin" id="isAdmin">
              <option value={true} selected={user.isAdmin}>Admin</option>
              <option value={false} selected={!user.isAdmin}>Regular User</option>
            </select>
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="isActive">Account Status</label>
            <select name="isActive" id="isActive">
              <option value={true} selected={user.isActive}>Active</option>
              <option value={false} selected={!user.isActive}>Inactive</option>
            </select>
          </div>
          
          <div className={styles.formGroupFull}>
            <label htmlFor="address">Address</label>
            <textarea id="address" name="address" placeholder={user.address || "Enter address"} rows="4"></textarea>
          </div>
          
          <button type="submit" className={styles.submitButton}>Update User</button>
        </form>
      </div>
    </div>
  );
};

export default SingleUserPage;
