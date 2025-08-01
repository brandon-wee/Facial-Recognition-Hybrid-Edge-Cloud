import Image from "next/image";
import Link from "next/link";
import styles from "@/app/ui/dashboard/models/models.module.css"; 
import Search from "@/app/ui/dashboard/search/search";
import Pagination from "@/app/ui/dashboard/pagination/pagination";
import { fetchModels } from "@/app/lib/data";
import { deleteModel } from "@/app/lib/actions";

const ModelsPage = async ({ searchParams }) => {
  const q = searchParams?.q || "";
  const page = searchParams?.page || 1;
  const { count, models } = await fetchModels(q, page);

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Model Management</h1>
        <p className={styles.subtitle}>Manage your machine learning models and their deployments</p>
      </div>
      
      <div className={styles.top}>
        <Search placeholder="Search for a model..." />
        <Link href="/dashboard/models/add">
          <button className={styles.addButton}>
            <span className={styles.addIcon}>+</span> Add New Model
          </button>
        </Link>
      </div>
      
      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Model Details</th>
              <th>Description</th>
              <th>Metrics</th>
              <th>Device</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.id}>
                <td>
                  <div className={styles.modelInfo}>
                    <Image
                      src={model.edgeDevice === "nicla" ? "/arduino-icon.png" : "/raspberry-icon.png"}
                      alt={model.edgeDevice === "nicla" ? "Arduino Nicla" : "Raspberry Pi"}
                      width={40}
                      height={40}
                      className={styles.modelImage}
                    />
                    <div className={styles.modelMeta}>
                      <span className={styles.modelTitle}>{model.title}</span>
                      <span className={`${styles.modelCategory} ${styles[model.cat]}`}>{model.cat}</span>
                    </div>
                  </div>
                </td>
                <td className={styles.descriptionCell}>
                  <div className={styles.truncateText}>{model.desc}</div>
                </td>
                <td>
                  <div className={styles.metrics}>
                    <div className={styles.metric}>
                      <span className={styles.metricLabel}>Accuracy:</span>
                      <span className={styles.metricValue}>{model.accuracy}%</span>
                    </div>
                    <div className={styles.metric}>
                      <span className={styles.metricLabel}>Response:</span>
                      <span className={styles.metricValue}>{model.responseTime} ms</span>
                    </div>
                    <div className={styles.metric}>
                      <span className={styles.metricLabel}>Size:</span>
                      <span className={styles.metricValue}>{model.size} MB</span>
                    </div>
                  </div>
                </td>
                <td>
                  <span className={`${styles.deviceTag} ${model.edgeDevice === "nicla" ? styles.nicla : styles.raspberry}`}>
                    {model.edgeDevice === "nicla" ? "Arduino Nicla" : "Raspberry Pi"}
                  </span>
                </td>
                <td>
                  <div className={styles.buttons}>
                    <Link href={`/dashboard/models/${model.id}`}>
                      <button className={`${styles.button} ${styles.view}`}>
                        View
                      </button>
                    </Link>
                    <form action={deleteModel}>
                      <input type="hidden" name="id" value={model.id} />
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

export default ModelsPage;