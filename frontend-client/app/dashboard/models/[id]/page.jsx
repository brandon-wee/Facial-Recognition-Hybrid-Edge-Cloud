import { updateModel } from "@/app/lib/actions"
import { fetchModel } from "@/app/lib/data"
import styles from "@/app/ui/dashboard/models/singleModel/singleModel.module.css"
import Image from "next/image"

const SingleModelPage = async ({ params }) => {
  const { id } = params
  const model = await fetchModel(id)

  return (
    <div className={styles.container}>
      <div className={styles.infoContainer}>
        <h2 className={styles.title}>Model Details</h2>
        <div className={styles.imgContainer}>
          <Image 
            src={model.edgeDevice === "nicla" ? "/arduino-icon.png" : "/raspberry-icon.png"}
            alt={model.edgeDevice === "nicla" ? "Arduino Nicla" : "Raspberry Pi"}
            fill 
            className={styles.deviceImage}
          />
        </div>
        <h1 className={styles.modelName}>{model.title}</h1>
        
        <div className={styles.detailsCard}>
          <div className={styles.detail}>
            <span className={styles.detailTitle}>Category:</span>
            <span className={`${styles.detailValue} ${styles.category} ${styles[model.cat]}`}>{model.cat}</span>
          </div>
          <div className={styles.detail}>
            <span className={styles.detailTitle}>Accuracy:</span>
            <span className={styles.detailValue}>{model.accuracy}%</span>
          </div>
          <div className={styles.detail}>
            <span className={styles.detailTitle}>Response Time:</span>
            <span className={styles.detailValue}>{model.responseTime} ms</span>
          </div>
          <div className={styles.detail}>
            <span className={styles.detailTitle}>Edge Device:</span>
            <span className={styles.detailValue}>{model.edgeDevice === "nicla" ? "Arduino Nicla" : "Raspberry Pi"}</span>
          </div>
          <div className={styles.detail}>
            <span className={styles.detailTitle}>Size:</span>
            <span className={styles.detailValue}>{model.size} MB</span>
          </div>
        </div>
      </div>
      
      <div className={styles.formContainer}>
        <h2 className={styles.title}>Edit Model</h2>
        <form action={updateModel} className={styles.form}>
          <input type="hidden" name="id" value={model.id} />
          
          <div className={styles.formGroup}>
            <label htmlFor="title">Title</label>
            <input 
              id="title"
              type="text" 
              name="title" 
              placeholder={model.title}
              required
            />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="cat">Category</label>
            <select id="cat" name="cat" required>
              <option value="">Select a category</option>
              <option value="base" selected={model.cat === "base"}>Base</option>
              <option value="distilled" selected={model.cat === "distilled"}>Distilled</option>
              <option value="quantized" selected={model.cat === "quantized"}>Quantized</option>
              <option value="distilled2" selected={model.cat === "distilled2"}>Distilled-Quantized</option>
            </select>
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="accuracy">Accuracy (%)</label>
            <input 
              id="accuracy"
              type="number" 
              name="accuracy" 
              placeholder={model.accuracy}
              step="0.01"
              min="0"
              max="100"
              required
            />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="responseTime">Response Time (ms)</label>
            <input
              id="responseTime"
              type="number"
              name="responseTime"
              placeholder={model.responseTime}
              min="0"
              required
            />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="size">Size (MB)</label>
            <input
              id="size"
              type="number"
              name="size"
              placeholder={model.size}
              min="0"
              required
            />
          </div>
          
          <div className={styles.formGroup}>
            <label htmlFor="edgeDevice">Edge Device</label>
            <select id="edgeDevice" name="edgeDevice" required>
              <option value="">Select an edge device</option>
              <option value="nicla" selected={model.edgeDevice === "nicla"}>Nicla Arduino</option>
              <option value="raspberry" selected={model.edgeDevice === "raspberry"}>Raspberry Pi Model B</option>
            </select>
          </div>
          
          <div className={styles.formGroupFull}>
            <label htmlFor="desc">Description</label>
            <textarea 
              id="desc"
              name="desc" 
              rows="8" 
              placeholder={model.desc}
              required
            ></textarea>
          </div>
          
          <button type="submit" className={styles.submitButton}>Update Model</button>
        </form>
      </div>
    </div>
  )
}

export default SingleModelPage