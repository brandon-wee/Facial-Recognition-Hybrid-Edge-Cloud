````markdown
# Face Recognition API

This repository contains a FastAPI-based Face Recognition API leveraging [InsightFace](https://github.com/deepinsight/insightface) for face embedding extraction and [MongoDB](https://www.mongodb.com/) for storing and updating user embeddings. The service allows you to:

- Load known face embeddings at startup.
- Accept base64‐encoded face crops and return recognized names (or “Unknown”).
- Log all recognition metadata to a `logs.txt` file.
- Upload multiple images for a given user, compute their average face embedding, and update MongoDB accordingly.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Environment Variables](#environment-variables)  
4. [Running the Server](#running-the-server)  
5. [Endpoints](#endpoints)  
6. [Project Structure](#project-structure)  
7. [Logging](#logging)  
8. [License](#license)  

---

## Prerequisites

- Python 3.8 or higher  
- A running MongoDB instance (Atlas, local, or any MongoDB-compatible service)  
- GPU (optional, but recommended for faster InsightFace inference)  
- `ffmpeg` or appropriate video codec (if you plan to extend the project to video streams)  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/face-recognition-api.git
   cd face-recognition-api
````

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   This project includes a `requirements.txt` listing all necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

   Example contents of `requirements.txt` (for reference):

   ```
   fastapi
   uvicorn[standard]
   python-dotenv
   pymongo[srv]
   numpy
   opencv-python
   insightface
   ```

   > **Note:** If you already have some packages installed system-wide, you can omit them from `requirements.txt`, but ensure versions remain compatible.

---

## Environment Variables

Create a file named `.env` in the project root (same level as `main.py` or `app.py`). The API expects at least one environment variable:

```dotenv
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/dashboard?retryWrites=true&w=majority
```

* `MONGO_URI`: Your MongoDB connection string.

  * For MongoDB Atlas, it usually looks like:

    ```
    MONGO_URI=mongodb+srv://myUsername:myPassword@cluster0.xyz.mongodb.net/dashboard?retryWrites=true&w=majority
    ```
  * If running a local instance:

    ```
    MONGO_URI=mongodb://localhost:27017/dashboard
    ```

Make sure **not** to commit your `.env` file to version control.

---

## Running the Server

By default, the FastAPI application file is named `main.py`. To start the server, run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

* `--reload` will auto‐restart the server on code changes (useful during development).
* You can change the host/port as needed.

Once running, you can visit the interactive OpenAPI docs at:

```
http://localhost:8000/docs
```

or the Redoc interface at:

```
http://localhost:8000/redoc
```

---

## Endpoints

Below is a summary of all available endpoints:

### 1. GET `/`

* **Description**: Simple HTML landing page.
* **Response**: A brief HTML snippet indicating usage (“Face Recognition API” and instructions).

### 2. POST `/recognize`

* **Description**: Recognize faces in one or more base64‐encoded crops.

* **Request Body**:

  ```json
  {
    "bboxes": [
      {
        "bbox": [x1, y1, x2, y2],
        "crop": "<base64-jpg-string>"
      },
      ...
    ],
    ... any other fields ...
  }
  ```

  * `bbox`: The bounding box coordinates of the face (`[x1, y1, x2, y2]`).
  * `crop`: A base64‐encoded JPEG image of the face crop.

* **Behavior**:

  1. Decode each `crop` into an OpenCV image (`cv2`).
  2. Run InsightFace on the crop to extract a normalized embedding.
  3. Compare against all known embeddings (loaded at startup).
  4. If cosine similarity ≥ `SIM_THRESHOLD` (default 0.3), return the corresponding username. Otherwise, label as `"Unknown <n>"`.
  5. Save the original base64 bytes to `photo/photo_<name>_<counter>.png` on disk.
  6. Build an enriched response entry containing:

     * `bbox`
     * original `crop` (base64 string)
     * `name` (recognized username or “Unknown …”)
     * `similarity` score (float)
  7. Overwrite the `"bboxes"` field in the request JSON with this enriched list.
  8. Append the entire enriched JSON to `logs.txt` (newline‐delimited).
  9. Return a JSON response:

     ```json
     {
       "status": "recognized",
       "meta": {
         // the enriched request JSON, including “bboxes” with name & similarity
       }
     }
     ```

* **Example**:

  ```bash
  curl -X POST "http://localhost:8000/recognize" \
       -H "Content-Type: application/json" \
       -d '{
         "bboxes": [
           {
             "bbox": [10, 20, 150, 200],
             "crop": "/9j/4AAQSkZJRgABAQAAAQABAAD/..."
           }
         ]
       }'
  ```

### 3. GET `/metadata`

* **Description**: Returns the most recently processed (enriched) recognition JSON.
* **Response**:

  ```json
  { /* contents of latest_meta – same structure as returned by /recognize */ }
  ```

### 4. POST `/calculate_average_embedding`

* **Description**: Upload multiple face images for a given `username`, compute the average embedding, update MongoDB, and return the new embedding vector.

* **Request**:

  * **Form Data**:

    * `username` (string): The user’s unique username in MongoDB.
    * `files` (List of image files): Each file (e.g., JPEG/PNG) should contain a frontal face.

* **Behavior**:

  1. Read each uploaded file into a NumPy array via OpenCV (`cv2.imdecode`).
  2. For each image, run InsightFace to extract the first face embedding.
  3. If at least one embedding is found, compute the element‐wise average and normalize to unit length.
  4. Call `update_embedding(username, avg_emb)`:

     * Fetch existing embedding array (if any) from the `users` collection.
     * If user exists and old/new dimensions match, compute a simple element-wise average of the old and new embedding. Otherwise, overwrite with `avg_emb`.
     * Update the MongoDB document’s `embeddings` field.
  5. Reload `known_embeddings` and `known_names` so that future `/recognize` calls include this new embedding.
  6. Return:

     ```json
     {
       "status": "success",
       "username": "<username>",
       "embedding": [ /* new averaged embedding as list */ ],
       "images": <number_of_images_processed>
     }
     ```
  7. If no faces were detected in any uploaded image, return:

     ```json
     {
       "status": "error",
       "message": "No faces detected in any image."
     }
     ```
  8. If `username` is not found in MongoDB, return:

     ```json
     {
       "status": "error",
       "message": "User '<username>' not found."
     }
     ```

* **Example**:

  ```bash
  curl -X POST "http://localhost:8000/calculate_average_embedding" \
       -F "username=johndoe" \
       -F "files=@/path/to/face1.jpg" \
       -F "files=@/path/to/face2.png"
  ```

---

## Project Structure

```
face-recognition-api/
├── .env
├── README.md
├── requirements.txt
├── main.py                   # (or app.py) FastAPI application code
├── photo/                    # Directory where cropped face images are saved
│   └── photo_<name>_<n>.png
├── logs.txt                  # Newline-delimited JSON of every recognition call
└── ... (any other modules you add)
```

* **`.env`**
  Contains sensitive environment variables like `MONGO_URI`.
* **`requirements.txt`**
  Lists all Python dependencies.
* **`main.py`**
  Entry point for the FastAPI app.
* **`photo/`**
  Stores face crop images (as `.png`) whenever a face is recognized (known or unknown).
* **`logs.txt`**
  Appends each enriched JSON payload from `/recognize` to a single text file (one JSON object per line).

---

## Logging

* **Recognition Logs**
  Every time `/recognize` is called, the enriched metadata (with names and similarity scores) is appended to `logs.txt`. You can use this log file to audit recognition requests, retrain embeddings, or track unknown faces.

* **Console Output**

  * At startup, the application prints all known usernames (fetched from MongoDB).
  * Whenever embeddings are updated via `/calculate_average_embedding`, you’ll see the updated `known_names` printed in the console.

---

## Tips & Troubleshooting

1. **InsightFace GPU/CPU**

   * By default, `face_app.prepare(ctx_id=0, det_size=(320, 320))` attempts to use the first GPU (CUDA device `0`).
   * If you do not have a GPU or wish to force CPU only, change `ctx_id=-1`.
   * Adjust `det_size=(h, w)` to trade off detection speed vs. accuracy.

2. **MongoDB Connection Errors**

   * Ensure `MONGO_URI` is valid and your IP is whitelisted (for MongoDB Atlas).
   * If using a local MongoDB instance, confirm it’s running on `mongodb://localhost:27017`.
   * Check that the database name (`dashboard`) and collection (`users`) exist, or adjust accordingly in `main.py`.

3. **“.env” File Not Loading**

   * Make sure `python-dotenv` is installed (`pip install python-dotenv`).
   * Verify that `.env` resides in the same directory where you run `uvicorn main:app`.

4. **Known Embeddings Array Shape**

   * If you see a `ValueError` when stacking embeddings, it’s likely because existing embeddings in the database have mismatched dimensions.
   * To fix, either delete problematic user documents or ensure all embeddings have the same length (e.g., 512 for InsightFace’s default ResNet models).

5. **Adjusting Similarity Threshold**

   * The default `SIM_THRESHOLD = 0.3` may be too lenient or strict depending on your dataset.
   * Experiment by printing `sim_val` for known users to find a suitable cutoff.

---

## License

This project is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute as needed.

---

> **Acknowledgments**
>
> * [InsightFace](https://github.com/deepinsight/insightface) for state-of-the-art face recognition models.
> * [FastAPI](https://fastapi.tiangolo.com/) for making asynchronous API development simple and fast.
> * [MongoDB](https://www.mongodb.com/) for flexible storage of embeddings.
> * [OpenCV](https://opencv.org/) for image decoding and basic preprocessing.

```
```
