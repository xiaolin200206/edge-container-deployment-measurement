#!/usr/bin/env python3
import time
import os
import sys
import csv
import cv2
import gc
import psutil
import subprocess
import numpy as np
import traceback
import requests
from datetime import datetime
import onnxruntime as ort

# ==========================================
# Config & Settings
# ==========================================
def get_absolute_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

MODEL_PATH = get_absolute_path("basil_mobilenet.onnx")

CYCLE_ACTIVE_SEC = 60
CYCLE_SLEEP_SEC = 15

SAVE_DATA_LOG = True
SAVE_IMAGES = True
SAVE_IMG_INTERVAL = 2.0
INFERENCE_SIZE = 224

TELEGRAM_BOT_TOKEN = "" #enter yourself token
TELEGRAM_CHAT_ID = "" #enter yourself id
TELEGRAM_COOLDOWN_SEC = 60.0

CLASSIFICATION_THRESHOLD = 0.70
MAX_TEMP_LIMIT = 82.0

# CRITICAL: Must match training label mapping
# Training used FixedSubset: Background=0, Basil_healthy=1, Disease=2
CLASS_NAMES = ["background", "basil_healthy", "disease"]

# ==========================================
# System Monitor Helper
# ==========================================
class SystemMonitor:
    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read()) / 1000.0
        except:
            return 0.0

    def get_throttled_state(self):
        try:
            output = subprocess.check_output(["vcgencmd", "get_throttled"]).decode()
            status_hex = output.split('=')[1].strip()
            status_int = int(status_hex, 16)
            if status_int > 0: return "Yes"
            return "No"
        except:
            return "Unknown"

    def get_ram_usage(self):
        return psutil.virtual_memory().percent

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=None)

# ==========================================
# Main System
# ==========================================
class BasilDataCollector:
    def __init__(self):
        print("Initializing Basil Edge-AI Alert System (ONNX + Telegram)...")
        self.monitor = SystemMonitor()
        self.session_start_time = datetime.now()

        self.last_telegram_time = 0.0
        self.history_buffer = []
        self.is_rgb_input = False  # Track camera color format

        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model not found at {MODEL_PATH}")
            sys.exit(1)

        print(f"Loading ONNX model: {os.path.basename(MODEL_PATH)}")
        try:
            self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            print("Warming up model...")
            dummy_input = np.zeros((1, 3, INFERENCE_SIZE, INFERENCE_SIZE), dtype=np.float32)
            self.session.run([self.output_name], {self.input_name: dummy_input})
            print(f"Ready. Classes: {CLASS_NAMES}")
            print(f"Confidence Threshold: {CLASSIFICATION_THRESHOLD}")
        except Exception as e:
            print(f"Model Load Failed: {e}")
            sys.exit(1)

        self.setup_logging()
        self.init_camera()

    def send_telegram_alert(self, img_path, message):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        try:
            with open(img_path, 'rb') as photo:
                payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message}
                files = {'photo': photo}
                response = requests.post(url, data=payload, files=files, timeout=5.0)
            if response.status_code == 200:
                print("Telegram Alert sent successfully!")
            else:
                print(f"Telegram sending failed: {response.text}")
        except Exception as e:
            print(f"Telegram Network Error: {e}")

    def setup_logging(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(script_dir, "basil_logs", f"session_{ts}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.img_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "basil_data.csv")
        self.event_log_path = os.path.join(self.log_dir, "cycle_events.csv")

        if SAVE_DATA_LOG:
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "Timestamp", "Latency_ms", "FPS",
                    "CPU_%", "RAM_%", "Temp_C", "Throttled",
                    "Predicted_Class", "Confidence"
                ])
            with open(self.event_log_path, 'w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Event", "Temp_C", "Note"])

    def log_event(self, event_name, note=""):
        if not SAVE_DATA_LOG: return
        temp = self.monitor.get_cpu_temp()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.event_log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ts, event_name, f"{temp:.1f}", note])

    def init_camera(self):
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"})
            self.picam2.configure(config)
            self.picam2.start()
            self.camera_type = "Picamera2"
            self.is_rgb_input = True  # Picamera2 outputs RGB
            print("Camera: Picamera2 (RGB format)")
        except ImportError:
            self.cap = cv2.VideoCapture(0)
            self.camera_type = "USB"
            self.is_rgb_input = False  # USB camera via OpenCV outputs BGR
            print("Camera: USB (BGR format)")

    def preprocess(self, frame):
        """Replicate PyTorch test_transform: Resize(256) -> CenterCrop(224) -> Normalize"""
        h, w = frame.shape[:2]
        # 1. Resize shortest side to 256, keep aspect ratio
        if h < w:
            new_h = 256
            new_w = int(w * (256 / h))
        else:
            new_w = 256
            new_h = int(h * (256 / w))
        img = cv2.resize(frame, (new_w, new_h))

        # 2. Center crop 224x224
        sy = (new_h - 224) // 2
        sx = (new_w - 224) // 2
        img = img[sy:sy+224, sx:sx+224]

        # 3. Convert to RGB if needed (model expects RGB)
        if not self.is_rgb_input:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # If Picamera2, frame is already RGB — no conversion needed

        # 4. Normalize with ImageNet stats
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, frame):
        input_tensor = self.preprocess(frame)
        t0 = time.time()
        raw_output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        latency = (time.time() - t0) * 1000

        logits = raw_output[0]
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / exp_preds.sum()

        best_class_idx = np.argmax(probs)
        best_class_name = CLASS_NAMES[best_class_idx]
        best_conf = float(probs[best_class_idx])

        # DEBUG - remove after testing
        print(f"logits: [BG:{logits[0]:.2f}, HL:{logits[1]:.2f}, DS:{logits[2]:.2f}] | "
              f"probs: [BG:{probs[0]:.2f}, HL:{probs[1]:.2f}, DS:{probs[2]:.2f}] | "
              f"pred: {best_class_name} {best_conf:.2f}")

        return best_class_name, best_conf, latency

    def run(self):
        print(f"STARTING CYCLE: {CYCLE_ACTIVE_SEC}s Active / {CYCLE_SLEEP_SEC}s Sleep")
        self.log_event("SYSTEM_START", f"Cycle: {CYCLE_ACTIVE_SEC}/{CYCLE_SLEEP_SEC}")

        cycle_start_time = time.time()
        is_active_mode = True
        self.log_event("CYCLE_ACTIVE_START")

        last_save = time.time()
        fps_start = time.time()
        fps_cnt = 0; fps = 0.0; frame_cnt = 0

        try:
            while True:
                current_time = time.time()
                cycle_elapsed = current_time - cycle_start_time

                if is_active_mode:
                    if cycle_elapsed > CYCLE_ACTIVE_SEC:
                        print("Entering Sleep Mode...")
                        self.log_event("CYCLE_SLEEP_START")
                        is_active_mode = False
                        cycle_start_time = current_time
                        cv2.destroyAllWindows()
                        continue
                else:
                    if cycle_elapsed > CYCLE_SLEEP_SEC:
                        print("Waking up! Active Mode.")
                        self.log_event("CYCLE_ACTIVE_START")
                        is_active_mode = True
                        cycle_start_time = current_time
                        fps_start = time.time()
                    else:
                        time.sleep(0.5)
                        continue

                temp = self.monitor.get_cpu_temp()
                if temp > MAX_TEMP_LIMIT:
                    print(f"OVERHEAT ({temp}C). Sleeping.")
                    time.sleep(5.0)
                    continue

                if frame_cnt % 200 == 0: gc.collect()

                if self.camera_type == "Picamera2":
                    frame = self.picam2.capture_array()  # RGB format
                else:
                    ret, frame = self.cap.read()  # BGR format
                    if not ret: break

                pred_class, conf, latency = self.infer(frame)

                # Anti-Jitter: Temporal smoothing over 5 frames
                if pred_class != "background" and conf >= CLASSIFICATION_THRESHOLD:
                    self.history_buffer.append(pred_class)
                else:
                    self.history_buffer.append("none")

                if len(self.history_buffer) > 5:
                    self.history_buffer.pop(0)

                # Require 3/5 consistent predictions to trigger
                display_class = "none"
                if self.history_buffer.count("disease") >= 3:
                    display_class = "disease"
                elif self.history_buffer.count("basil_healthy") >= 3:
                    display_class = "basil_healthy"

                cpu = self.monitor.get_cpu_usage()
                ram = self.monitor.get_ram_usage()

                # Convert to BGR for OpenCV display and saving
                if self.is_rgb_input:
                    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    vis = frame.copy()

                if display_class != "none":
                    if display_class == "disease":
                        color = (0, 0, 255)  # Red in BGR
                        label = f"DISEASE: {conf:.0%}"
                    else:
                        color = (0, 255, 0)  # Green in BGR
                        label = f"HEALTHY: {conf:.0%}"

                    cv2.rectangle(vis, (0, 0), (vis.shape[1], vis.shape[0]), color, 10)
                    cv2.putText(vis, label, (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

                fps_cnt += 1
                if fps_cnt >= 10:
                    fps = 10 / (time.time() - fps_start)
                    fps_start = time.time()
                    fps_cnt = 0

                status = f"FPS:{fps:.1f} | T:{temp:.0f}C | L:{latency:.1f}ms"
                cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Basil Classifier", cv2.resize(vis, (640, 480)))

                # Data logging
                if SAVE_DATA_LOG:
                    with open(self.csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            datetime.now().strftime("%H:%M:%S.%f")[:-3],
                            f"{latency:.1f}", f"{fps:.1f}",
                            f"{cpu:.1f}", f"{ram:.1f}", f"{temp:.1f}",
                            self.monitor.get_throttled_state(),
                            pred_class, f"{conf:.2f}"
                        ])

                # Save images + Telegram alert (vis is BGR, correct for imwrite)
                if SAVE_IMAGES and display_class != "none" and (time.time() - last_save > SAVE_IMG_INTERVAL):
                    img_name = f"{display_class}_{frame_cnt}.jpg"
                    img_path = os.path.join(self.img_dir, img_name)
                    cv2.imwrite(img_path, vis)
                    last_save = time.time()

                    if display_class == "disease" and (time.time() - self.last_telegram_time > TELEGRAM_COOLDOWN_SEC):
                        alert_msg = (
                            f"🚨 Greenhouse Alert: Basil Disease Detected!\n"
                            f"🌡️ System Temp: {temp:.1f}°C\n"
                            f"🧠 AI Confidence: {conf:.0%}\n"
                            f"⏱️ Inference Latency: {latency:.1f}ms\n"
                            f"📊 Threshold: {CLASSIFICATION_THRESHOLD}"
                        )
                        self.send_telegram_alert(img_path, alert_msg)
                        self.last_telegram_time = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'): break
                frame_cnt += 1

        except KeyboardInterrupt:
            print("Stopped.")
        except Exception as e:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            try:
                if self.camera_type == "Picamera2": self.picam2.stop()
                else: self.cap.release()
            except: pass

if __name__ == "__main__":
    app = BasilDataCollector()
    app.run()
