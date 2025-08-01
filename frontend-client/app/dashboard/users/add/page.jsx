import { addUser } from "@/app/lib/actions";
import styles from "@/app/ui/dashboard/users/addUser/addUser.module.css";

const AddUserPage = () => {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Add New User</h1>
      <form action={addUser} className={styles.form}>
        <div className={styles.formGroup}>
          <label htmlFor="username">Username</label>
          <input type="text" id="username" name="username" required />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="email">Email</label>
          <input type="email" id="email" name="email" required />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="password">Password</label>
          <input type="password" id="password" name="password" required />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="phone">Phone Number</label>
          <input type="tel" id="phone" name="phone" />
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="isAdmin">Admin Status</label>
          <select name="isAdmin" id="isAdmin">
            <option value="" disabled>
              Select admin status
            </option>
            <option value={true}>Admin</option>
            <option value={false} selected>
              Regular User
            </option>
          </select>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="isActive">Account Status</label>
          <select name="isActive" id="isActive">
            <option value="" disabled>
              Select account status
            </option>
            <option value={true} selected>
              Active
            </option>
            <option value={false}>Inactive</option>
          </select>
        </div>

        <div className={styles.formGroupFull}>
          <label htmlFor="address">Address</label>
          <textarea
            name="address"
            id="address"
            rows="6"
            placeholder="Enter user address"
          ></textarea>
        </div>

        <button type="submit" className={styles.submitButton}>
          Add User
        </button>
      </form>
    </div>
  );
};

export default AddUserPage;
