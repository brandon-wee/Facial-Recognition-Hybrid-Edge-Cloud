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

    # — raw‐socket POST —
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