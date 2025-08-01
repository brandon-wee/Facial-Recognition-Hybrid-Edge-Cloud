import { addModel } from "@/app/lib/actions"
import styles from "@/app/ui/dashboard/models/addModel/addModel.module.css"

const AddModelPage = () => {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Add New Model</h1>
      <form action={addModel} className={styles.form}>
        <div className={styles.formGroup}>
          <label htmlFor="title">Model Title</label>
          <input 
            type="text" 
            id="title" 
            name="title" 
            placeholder="Enter model title" 
            required 
          />
        </div>
        
        <div className={styles.formGroup}>
          <label htmlFor="cat">Category</label>
          <select id="cat" name="cat" required>
            <option value="">Select model category</option>
            <option value="base">Base</option>
            <option value="distilled">Distilled</option>
            <option value="quantized">Quantized</option>
            <option value="distilled2">Distilled-Quantized</option>
          </select>
        </div>
        
        <div className={styles.formGroup}>
          <label htmlFor="accuracy">Accuracy (%)</label>
          <input 
            type="number" 
            id="accuracy"
            placeholder="Enter model accuracy"
            name="accuracy"
            step="0.01" 
            min="0"
            max="100"
            required
          />
        </div>
        
        <div className={styles.formGroup}>
          <label htmlFor="responseTime">Response Time (ms)</label>
          <input
            type="number"
            id="responseTime"
            placeholder="Enter response time"
            name="responseTime"
            min="0"
            required
          />
        </div>
        
        <div className={styles.formGroup}>
          <label htmlFor="size">Size (MB)</label>
          <input
            type="number"
            id="size"
            placeholder="Enter model size"
            name="size"
            min="0"
            required
          />
        </div>
        
        <div className={styles.formGroup}>
          <label htmlFor="edgeDevice">Edge Device</label>
          <select id="edgeDevice" name="edgeDevice" required>
            <option value="">Select edge device</option>
            <option value="nicla">Nicla Arduino</option>
            <option value="raspberry">Raspberry Pi Model B</option>
          </select>
        </div>
        
        <div className={styles.formGroupFull}>
          <label htmlFor="desc">Description</label>
          <textarea
            id="desc"
            name="desc"
            rows="8"
            placeholder="Enter model description"
            required
          ></textarea>
        </div>
        
        <button type="submit" className={styles.submitButton}>Add Model</button>
      </form>
    </div>
  )
}

export default AddModelPage