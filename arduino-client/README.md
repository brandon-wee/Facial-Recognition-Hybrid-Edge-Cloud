````markdown
# Arduino Nicla Vision Micropython Client

This repository contains a Micropython script designed to run on the Arduino Nicla Vision (OpenMV-based) board. The script:

- Connects to a specified Wi-Fi network.
- Captures frames from the onboard camera.
- Runs a FOMO-based face detection model (64×64 input) locally.
- Extracts bounding boxes and JPEG-compressed face crops (base64-encoded).
- Sends JSON payloads over raw TCP to a backend “/recognize” endpoint whenever faces are detected.
- Prints FPS and detection results to the serial console.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Download & Install OpenMV IDE](#download--install-openmv-ide)  
3. [Flashing Micropython & Uploading the Script](#flashing-micropython--uploading-the-script)  
4. [Configuration](#configuration)  
5. [How It Works](#how-it-works)  
6. [Usage](#usage)  
7. [Troubleshooting & Tips](#troubleshooting--tips)  
8. [Project Structure](#project-structure)  
9. [License](#license)  

---

## Prerequisites

1. **Hardware**  
   - **Arduino Nicla Vision** board (configured with the VisionShield and powered via USB).  
   - USB cable to connect Nicla Vision to your development computer.  
   - A Wi-Fi network (2.4 GHz) with known SSID and password.  
   - A running backend server (e.g., FastAPI `/recognize` endpoint) listening on a reachable IP and port.

2. **Software**  
   - **OpenMV IDE** (for uploading Micropython scripts to Nicla Vision).  
   - **Micropython firmware** for Nicla Vision (the OpenMV-flavor firmware).  
   - A computer (Windows, macOS, or Linux) with USB ports.  

3. **Backend Requirements**  
   - The backend must accept HTTP POST at:
     ```
     http://<SERVER_HOST>:<SERVER_PORT>/recognize
     ```
     and process JSON payloads of the form:
     ```json
     {
       "fps": <float>,
       "people_count": <int>,
       "bboxes": [
         { "bbox": [x1, y1, x2, y2],  "crop": "<base64-jpg-string>" },
         …
       ]
     }
     ```
   - Make sure the backend’s IP/hostname is reachable from the Nicla Vision’s Wi-Fi network.

---

## Download & Install OpenMV IDE

1. **Visit the OpenMV Download Page**  
   - Open your browser and navigate to:  
     > https://openmv.io/pages/download  

2. **Choose Your OS**  
   - Download the installer for your operating system (Windows, macOS, or Linux).  

3. **Install OpenMV IDE**  
   - Run the downloaded installer and follow the on-screen instructions.  
   - After installation, launch **OpenMV IDE**.  

4. **Verify Connectivity**  
   - Connect your Nicla Vision board to a USB port.  
   - In OpenMV IDE, select the correct COM port (Windows) or `/dev/ttyACM0` (Linux/Mac).  
   - You should see a live serial console at the bottom of the IDE.  

---

## Flashing Micropython & Uploading the Script

1. **Obtain the Official Nicla Vision Micropython Firmware**  
   - Go to the Arduino Nicla Vision product pages or the OpenMV repository to find the latest `.bin` or `.fw` file for Nicla Vision.  
   - For example, you might find “`openmv-nicla-vision-vX.X.X.bin`” on the Arduino or OpenMV downloads page.

2. **Flash the Firmware**  
   - In **OpenMV IDE**, click on the **“Tools → Open Bootloader”** menu (or press the “Boot” button on the board if required).  
   - Click the **“Firmware Update”** button (the down arrow icon).  
   - Select the downloaded Micropython firmware file, and confirm.  
   - Wait until flashing completes. The board will reboot automatically into Micropython.

3. **Create or Open `main.py`**  
   - In OpenMV IDE, click **“File → New”** and paste the Micropython script (shown below) into the editor.  
   - Save it as `main.py` (this file name ensures it runs automatically on boot).

   ```python
   import network, socket, time, gc
   import sensor, image, math, ubinascii, ml, ujson as json
   from ml.utils import NMS

   # ————— CONFIG —————
   SSID           = "<WIFI-SSID>"
   PASSWORD       = "<WIFI-PASSWORD>"
   SERVER_HOST    = "<your-end-point>"
   SERVER_PORT    = 8000
   SERVER_PATH    = "/recognize"
   JPEG_QUALITY   = 25     # lower quality → smaller buffers
   CROP_SIZE      = (64,64)# downscale all face crops to 64×64
   FRAME_DELAY    = 100    # ms between frames
   MIN_CONFIDENCE = 0.4
   # ——————————————————

   # Wi-Fi
   wlan = network.WLAN(network.STA_IF)
   wlan.active(True)
   wlan.connect(SSID, PASSWORD)
   print("Connecting…")
   while not wlan.isconnected():
       time.sleep_ms(200)
       print(wlan)
   print("Wi-Fi OK, IP =", wlan.ifconfig()[0])

   # Cam + model
   sensor.reset()
   sensor.set_pixformat(sensor.RGB565)
   sensor.set_framesize(sensor.B64X64)
   sensor.set_windowing((240,240))
   # sensor.skip_frames(2000)

   model = ml.Model("fomo_face_detection")
   print("Loaded:", model)

   thr_val = math.ceil(MIN_CONFIDENCE * 255)
   threshold_list = [(thr_val, 255)]

   def fomo_post_process(mdl, inputs, outputs):
       n, oh, ow, oc = mdl.output_shape[0]
       nms = NMS(ow, oh, inputs[0].roi)
       for i in range(oc):
           heat = image.Image(outputs[0][0,:,:,i] * 255)
           blobs = heat.find_blobs(threshold_list, area_threshold=1, pixels_threshold=1)
           for b in blobs:
               x,y,w,h = b.rect()
               score = heat.get_statistics(thresholds=threshold_list, roi=b.rect()).l_mean()/255
               nms.add_bounding_box(x, y, x+w, y+h, score, i)
       return nms.get_bounding_boxes()

   clock = time.clock()

   while True:
       clock.tick()
       frame = sensor.snapshot()

       dets = model.predict([frame], callback=fomo_post_process)
       if len(dets) != 2:
           continue

       print(dets)
       dets = dets[1]
       bboxes = []

       for (x1,y1,x2,y2), score in dets:
           print(score, x1, y1, x2, y2)
           if score < MIN_CONFIDENCE: continue

           # — downscale crop to save RAM —
           # crop = frame.copy(roi=(x1, y1, x2, y2))
           # crop = crop.resize(*CROP_SIZE)
           copy = frame.copy()
           buf = copy.compress(quality=JPEG_QUALITY)
           b64 = ubinascii.b2a_base64(buf).decode().strip()
           bboxes.append({"bbox":[x1,y1,x2,y2],"crop":b64})

           frame.draw_rectangle(x1,y1, x2 + x1, y2 + y1, color=(255,0,0))

       payload = json.dumps({
           "fps": round(clock.fps(),2),
           "people_count": len(bboxes),
           "bboxes": bboxes
       })

       # — raw-socket POST —
       if len(bboxes) > 0:
           try:
               s = socket.socket()
               s.connect((SERVER_HOST, SERVER_PORT))
               req = (
                   "POST {} HTTP/1.1\r\n"
                   "Host: {}:{}\r\n"
                   "Content-Type: application/json\r\n"
                   "Content-Length: {}\r\n\r\n"
                   "{}"
               ).format(SERVER_PATH, SERVER_HOST, SERVER_PORT, len(payload), payload)
               s.send(req)
               print("→", s.recv(128))
           except Exception as e:
               print("Post err:", e)
           finally:
               s.close()

       gc.collect()               # free any leftover
       time.sleep_ms(FRAME_DELAY)
````

4. **Install the FOMO Face Detection Model**

   * In OpenMV IDE’s left sidebar, expand **“Site Packages”** or **“Internal Flash”**.
   * Click **“Tools → Package Manager”**, search for `fomo_face_detection`, and install it.
   * Alternatively, copy `fomo_face_detection.py` into the board’s `/flash/lib/ml` folder if you have a local `.py` model file.

5. **Save & Run**

   * Click **“Save”** (floppy icon) in OpenMV IDE to write `main.py` to the Nicla Vision’s flash.
   * Press the **“Play”** button (green arrow) to restart and run the script.
   * Observe the serial console for “Connecting…”, “Wi-Fi OK, IP = …”, and “Loaded: fomo\_face\_detection”.

---

## Configuration

Before saving `main.py`, update the following constants at the top of the script:

```python
SSID           = "<WIFI-SSID>"           # Your Wi-Fi network name
PASSWORD       = "<WIFI-PASSWORD>"       # Your Wi-Fi password
SERVER_HOST    = "<your-end-point>"      # e.g., "192.168.1.100"
SERVER_PORT    = 8000                    # Backend port (default 8000)
SERVER_PATH    = "/recognize"            # Endpoint path on your server
JPEG_QUALITY   = 25                      # 10–30 is a good tradeoff on Nicla Vision
CROP_SIZE      = (64,64)                 # Crop resizing before base64 (optional)
FRAME_DELAY    = 100                     # Milliseconds between frames
MIN_CONFIDENCE = 0.4                     # FOMO confidence threshold (0.0–1.0)
```

* **Wi-Fi Credentials**: Ensure the Nicla Vision can join your local network.
* **SERVER\_HOST**: IP address or hostname of your backend (must be reachable from Nicla Vision).
* **SERVER\_PORT & SERVER\_PATH**: Match your backend’s listening port and route.

---

## How It Works

1. **Wi-Fi Connection**

   * The script configures the Nicla Vision’s STA interface and blocks until it successfully connects.
   * Prints its assigned IP to the serial console once connected.

2. **Camera & Model Setup**

   * Initializes the camera sensor in RGB565 format, `64×64` frame size, with a central 240×240 window.
   * Loads a pretrained FOMO face detection model (`fomo_face_detection.tflite`) from the onboard filesystem.

3. **Face Detection Loop**

   * On each loop iteration:

     1. Capture a snapshot (`sensor.snapshot()`).
     2. Run `model.predict([frame], callback=fomo_post_process)`:

        * The callback `fomo_post_process` applies non-maximum suppression (NMS) on raw heatmap outputs to generate bounding boxes.
     3. For each detected face with score ≥ `MIN_CONFIDENCE`:

        * Draws a red rectangle on the live frame for debugging/preview.
        * Compresses the entire frame or just the crop to JPEG at `JPEG_QUALITY` to conserve memory.
        * Encodes the compressed bytes into base64 (`ubinascii.b2a_base64`).
        * Appends an object `{ "bbox": [x1,y1,x2,y2], "crop": "<base64>" }` to the `bboxes` list.
     4. Builds a JSON string containing:

        ```json
        {
          "fps": <measured FPS>,
          "people_count": <number of detected faces>,
          "bboxes": [ … ]
        }
        ```
     5. If any faces are detected (`len(bboxes)>0`), the script opens a raw TCP socket to `(SERVER_HOST, SERVER_PORT)`, formats an HTTP POST request (including `Content-Length`), sends it, prints the server’s response, and closes the socket.

4. **Garbage Collection & Delay**

   * Calls `gc.collect()` each loop to free unused memory (critical on constrained hardware).
   * Sleeps for `FRAME_DELAY` milliseconds to regulate processing rates.

---

## Usage

1. **Power & Boot**

   * Power the Nicla Vision via USB. The device will automatically run `main.py` on boot.
   * Open the **OpenMV IDE** serial console to monitor logs:

     ```
     Connecting…
     <network.WLAN object>
     …
     Wi-Fi OK, IP = 192.168.1.42
     Loaded: <fomo_face_detection model>
     ```
   * The board will continuously:

     * Detect faces in the 240×240 view.
     * Print detections (scores & coordinates) over USB serial.
     * Send JSON payloads to your backend whenever faces are found.

2. **Backend Integration**

   * Verify your backend’s console/logs to confirm it is receiving POST requests in the proper format.
   * Example payload:

     ```json
     {
       "fps": 7.85,
       "people_count": 1,
       "bboxes": [
         {
           "bbox": [50, 60, 100, 100],
           "crop": "/9j/4AAQSkZJRgABAQAAAQABAAD/…"
         }
       ]
     }
     ```

3. **Stopping the Script**

   * Remove power (disconnect USB), or upload a different script to override `main.py`.
   * You can also reset the Nicla Vision via the onboard reset button.

---

## Troubleshooting & Tips

1. **Cannot Connect to Wi-Fi**

   * Check that `SSID` and `PASSWORD` are correct.
   * Ensure you are in range of the 2.4 GHz Wi-Fi network (Nicla Vision does not support 5 GHz).
   * Use a simple network (no captive portal). Serial console will print repeated WLAN object lines until connected.

2. **Model Not Found / “Loaded:” Error**

   * Confirm you installed `fomo_face_detection` via OpenMV IDE’s Package Manager.
   * In the **File Browser** (OpenMV IDE), expand `/flash/lib/ml` and verify `fomo_face_detection.py` (or `.tflite`) is present.
   * If not, download it from the OpenMV model repository and place it under `/flash/lib/ml`.

3. **Out of Memory (OOM)**

   * Lower `JPEG_QUALITY` (e.g., 15 or 10).
   * Reduce `FRAME_DELAY` (increase the delay—for example, `200 ms`) so the board has more time to garbage collect.
   * Decrease `CROP_SIZE` or skip cropping (send full frames less frequently).

4. **No HTTP Response or Timeout**

   * Verify `SERVER_HOST` and `SERVER_PORT` are correct and reachable.
   * Ensure there are no firewall rules blocking TCP port 8000.
   * Temporarily adjust the backend to print raw TCP request contents to confirm connectivity.

5. **FPS Is Too Low**

   * Increase `FRAME_DELAY` (i.e., sleep longer between frames).
   * Disable rectangle drawing (`frame.draw_rectangle`) to save CPU cycles.
   * Reduce model complexity or switch to a smaller face detection model if available.

6. **Serial Console Flood**

   * The script prints every detection to serial. To reduce console spam, comment out `print(score, x1, y1, x2, y2)` or limit prints to once per second.

---

## Project Structure

```
micropython-nicla-vision-client/
├── main.py           # Micropython script (paste into OpenMV IDE)
├── README.md         # This file
└── LICENSE           # (Optional) License file if you choose to include one
```

* **`main.py`**
  Contains all code required for Wi-Fi setup, camera initialization, FOMO face detection, post-processing, JSON serialization, and raw TCP POST logic.

* **`README.md`**
  Describes setup, configuration, and usage instructions.

---

## License

This project is released under the [MIT License](LICENSE).

---

```
```
