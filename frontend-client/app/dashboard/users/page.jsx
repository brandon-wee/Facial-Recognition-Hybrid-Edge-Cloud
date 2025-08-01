import { deleteUser } from "@/app/lib/actions";
import { fetchUsers } from "@/app/lib/data";
import Pagination from "@/app/ui/dashboard/pagination/pagination";
import Search from "@/app/ui/dashboard/search/search";
import styles from "@/app/ui/dashboard/users/users.module.css";
import Image from "next/image";
import Link from "next/link";

const UsersPage = async ({ searchParams }) => {
  const q = searchParams?.q || "";
  const page = searchParams?.page || 1;
  const { count, users } = await fetchUsers(q, page);

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>User Management</h1>
        <p className={styles.subtitle}>Manage system users and their permissions</p>
      </div>
      
      <div className={styles.top}>
        <Search placeholder="Search for a user..." />
        <Link href="/dashboard/users/add">
          <button className={styles.addButton}>
            <span className={styles.addIcon}>+</span> Add New User
          </button>
        </Link>
      </div>
      
      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>User</th>
              <th>Email</th>
              <th>Created At</th>
              <th>Role</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.id}>
                <td>
                  <div className={styles.user}>
                    <Image
                      src={user.img || "/noavatar.png"}
                      alt={`${user.username}'s profile`}
                      width={40}
                      height={40}
                      className={styles.userImage}
                    />
                    <span className={styles.userName}>{user.username}</span>
                  </div>
                </td>
                <td className={styles.emailCell}>{user.email}</td>
                <td>{user.createdAt?.toString().slice(4, 16)}</td>
                <td>
                  <span className={`${styles.roleTag} ${user.isAdmin ? styles.adminRole : styles.clientRole}`}>
                    {user.isAdmin ? "Admin" : "Client"}
                  </span>
                </td>
                <td>
                  <span className={`${styles.statusIndicator} ${user.isActive ? styles.activeStatus : styles.inactiveStatus}`}>
                    {user.isActive ? "Active" : "Inactive"}
                  </span>
                </td>
                <td>
                  <div className={styles.buttons}>
                    <Link href={`/dashboard/users/${user.id}`}>
                      <button className={`${styles.button} ${styles.view}`}>
                        View
                      </button>
                    </Link>
                    <form action={deleteUser}>
                      <input type="hidden" name="id" value={(user.id)} />
                      <button className={`${styles.button} ${styles.delete}`}>
                        Delete
                      </button>
                    </form>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className={styles.paginationWrapper}>
        <Pagination count={count} />
      </div>
    </div>
  );
};

export default UsersPage;
