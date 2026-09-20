#!/usr/bin/env python3
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
import time
import sys
import csv
import cv2
import gc
import psutil
import numpy as np
import traceback
import requests
import threading
from collections import deque
from datetime import datetime
import onnxruntime as ort

# ==========================================
# Configuration
# ==========================================
def get_absolute_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

MODEL_PATH = get_absolute_path("MobileNetV2.onnx")

CYCLE_ACTIVE_SEC = 180
CYCLE_SLEEP_SEC = 45

SAVE_DATA_LOG = True
SAVE_IMAGES = True
SAVE_IMG_INTERVAL = 2.0

INFERENCE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50

# Temporal smoothing: majority vote over a sliding window
TEMPORAL_WINDOW = 5
TEMPORAL_MIN_VOTES = 3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Telegram credentials are read from environment variables, never hardcoded.
# Set them before running, e.g.:
#   export TELEGRAM_BOT_TOKEN="your_bot_token"
#   export TELEGRAM_CHAT_ID="your_chat_id"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_COOLDOWN_SEC = 60.0

MAX_TEMP_LIMIT = 82.0

# Class index order must match the training dataset's class_idx assignment:
# ['Background', 'Healthy', 'Disease'] -> 0, 1, 2
CLASS_NAMES = {
    0: "Background",
    1: "Healthy",
    2: "Disease",
}

ALERT_CLASS = "Disease"

CLASS_COLORS = {
    "Background": (128, 128, 128),
    "Healthy":    (0, 255, 0),
    "Disease":    (0, 0, 255),
}


class SystemMonitor:
    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except Exception:
            return 0.0

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=None)

    def get_ram_usage(self):
        return psutil.virtual_memory().percent


class BasilClassifier:
    def __init__(self):
        print("=" * 50)
        print("Basil Edge-AI Classification System (MobileNetV2 + ONNX Runtime)")
        print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Temporal Window: {TEMPORAL_WINDOW} frames, min votes: {TEMPORAL_MIN_VOTES}")
        print(f"Cycle: {CYCLE_ACTIVE_SEC}s active / {CYCLE_SLEEP_SEC}s sleep")
        print("=" * 50)

        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("Warning: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set. Alerts will be disabled.")

        self.monitor = SystemMonitor()
        self.last_telegram_time = 0.0
        self.recent_predictions = deque(maxlen=TEMPORAL_WINDOW)
        self.last_confirmed_class = None

        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            sys.exit(1)

        print("Loading MobileNetV2 ONNX...")
        self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warm up once so the first real frame isn't penalized by lazy init
        dummy = np.zeros((1, 3, INFERENCE_SIZE, INFERENCE_SIZE), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        print(f"Classes: {CLASS_NAMES}")
        print("Ready.")

        self.setup_logging()
        self.init_camera()

    def send_telegram_alert_thread(self, img_path, message):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return

        def send():
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            try:
                with open(img_path, 'rb') as photo:
                    payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message}
                    response = requests.post(url, data=payload, files={'photo': photo}, timeout=15.0)
                if response.status_code == 200:
                    print("Telegram: Alert sent!")
                else:
                    print(f"Telegram: Failed ({response.status_code})")
            except Exception as e:
                print(f"Telegram: Error {e}")
        threading.Thread(target=send, daemon=True).start()

    def setup_logging(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(script_dir, "basil_logs", f"session_{ts}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.img_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"basil_classification_data_{ts}.csv")
        if SAVE_DATA_LOG:
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "Timestamp", "Predicted_Class", "Confidence", "Confirmed_Class",
                    "Latency_ms", "FPS", "CPU_%", "RAM_%", "Temp_C"
                ])
            print(f"Logging to: {self.csv_path}")

    def init_camera(self):
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            ))
            self.picam2.start()
            time.sleep(2.0)  # let auto-exposure/white-balance settle
            try:
                self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                print("Camera: Autofocus enabled")
            except Exception as e:
                print(f"Camera: Autofocus not available ({e})")
            time.sleep(1.0)
            self.camera_type = "Picamera2"
            print("Camera: Picamera2 (RGB)")
        except ImportError:
            self.cap = cv2.VideoCapture(0)
            self.camera_type = "USB"
            print("Camera: USB (BGR)")

    def preprocess(self, frame):
        """frame: HxWx3, RGB (Picamera2) or BGR (USB). Returns a (1,3,224,224) normalized tensor."""
        if self.camera_type == "USB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def infer(self, frame):
        t0 = time.time()
        input_tensor = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        latency = (time.time() - t0) * 1000

        logits = outputs[0][0]
        probs = self.softmax(logits)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_class = CLASS_NAMES.get(pred_idx, f"class_{pred_idx}")

        return predicted_class, confidence, latency

    def update_temporal_smoothing(self, predicted_class, confidence):
        """Majority vote over the last N frames. Low-confidence frames count as 'unconfirmed'."""
        if confidence >= CONFIDENCE_THRESHOLD:
            self.recent_predictions.append(predicted_class)
        else:
            self.recent_predictions.append("unconfirmed")

        if len(self.recent_predictions) < TEMPORAL_WINDOW:
            return None

        votes = {}
        for cls in self.recent_predictions:
            votes[cls] = votes.get(cls, 0) + 1

        top_class = max(votes, key=votes.get)
        if top_class != "unconfirmed" and votes[top_class] >= TEMPORAL_MIN_VOTES:
            return top_class
        return None

    def run(self):
        print(f"STARTING: {CYCLE_ACTIVE_SEC}s active / {CYCLE_SLEEP_SEC}s sleep")
        print("Press 'q' to quit")

        cycle_start = time.time()
        is_active = True
        fps_start = time.time()
        fps_cnt = 0
        fps = 0.0
        frame_cnt = 0
        last_save = time.time()

        try:
            while True:
                now = time.time()
                elapsed = now - cycle_start

                if is_active:
                    if elapsed > CYCLE_ACTIVE_SEC:
                        print("Sleep mode...")
                        is_active = False
                        cycle_start = now
                        continue
                else:
                    if elapsed > CYCLE_SLEEP_SEC:
                        print("Active mode.")
                        is_active = True
                        cycle_start = now
                        fps_start = time.time()
                    else:
                        time.sleep(0.5)
                        continue

                temp = self.monitor.get_cpu_temp()
                if temp > MAX_TEMP_LIMIT:
                    print(f"OVERHEAT ({temp}C)")
                    time.sleep(5.0)
                    continue

                if frame_cnt % 200 == 0:
                    gc.collect()

                if self.camera_type == "Picamera2":
                    frame = self.picam2.capture_array()
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                predicted_class, confidence, latency = self.infer(frame)
                confirmed_class = self.update_temporal_smoothing(predicted_class, confidence)

                # No bounding boxes for classification - overlay text on the full frame
                vis = frame.copy()
                color = CLASS_COLORS.get(predicted_class, (255, 255, 255))
                cv2.putText(vis, f"{predicted_class} ({confidence*100:.1f}%)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if confirmed_class:
                    cv2.putText(vis, f"CONFIRMED: {confirmed_class}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                fps_cnt += 1
                if fps_cnt >= 10:
                    fps = 10 / (time.time() - fps_start)
                    fps_start = time.time()
                    fps_cnt = 0

                cpu = self.monitor.get_cpu_usage()
                ram = self.monitor.get_ram_usage()
                cv2.putText(vis, f"FPS:{fps:.1f} | Lat:{latency:.0f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"T:{temp:.1f}C | CPU:{cpu:.0f}% | RAM:{ram:.0f}%",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Run with `DISPLAY=:0 python3 basil_classifier.py` to show this on the Pi's HDMI output
                cv2.imshow("Basil Classifier", cv2.resize(vis, (800, 600)))

                if SAVE_DATA_LOG:
                    with open(self.csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            predicted_class, f"{confidence:.4f}",
                            confirmed_class if confirmed_class else "None",
                            f"{latency:.1f}", f"{fps:.1f}",
                            f"{cpu:.1f}", f"{ram:.1f}", f"{temp:.1f}"
                        ])

                if SAVE_IMAGES and confirmed_class == ALERT_CLASS and (now - last_save > SAVE_IMG_INTERVAL):
                    raw_path = os.path.join(self.img_dir, f"raw_{frame_cnt}.jpg")
                    if self.camera_type == "Picamera2":
                        cv2.imwrite(raw_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(raw_path, frame)

                    img_path = os.path.join(self.img_dir, f"det_{frame_cnt}.jpg")
                    if self.camera_type == "Picamera2":
                        cv2.imwrite(img_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(img_path, vis)
                    last_save = now

                    if now - self.last_telegram_time > TELEGRAM_COOLDOWN_SEC:
                        alert_msg = (
                            f"Basil Disease Detected\n\n"
                            f"Confirmed: {confirmed_class}\n"
                            f"Confidence: {confidence*100:.1f}%\n\n"
                            f"Temp: {temp:.1f}C\n"
                            f"Latency: {latency:.1f}ms"
                        )
                        self.send_telegram_alert_thread(img_path, alert_msg)
                        self.last_telegram_time = now

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_cnt += 1

        except KeyboardInterrupt:
            print("Stopped.")
        except Exception:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            try:
                if self.camera_type == "Picamera2":
                    self.picam2.stop()
                else:
                    self.cap.release()
            except Exception:
                pass


if __name__ == "__main__":
    app = BasilClassifier()
    app.run()
