# Basil Edge-AI: Real-Time Disease Classification on Raspberry Pi 5

## 📌 Overview

This repository provides a complete, production-ready **Standard Operating Procedure (SOP)** for deploying the **Basil Disease Classification System** on a compute-constrained edge device (**Raspberry Pi 5**).

Designed for real-world agricultural environments, this deployment is:

- **Strictly headless** (no GUI)
- **Highly stable**
- **Auto-starting on boot** (recovers after power loss)
- Optimized for **zero-touch maintenance** for greenhouse operators

---

## 🛠 Prerequisites

### Hardware

- **Board:** Raspberry Pi 5 (⭐ 8GB RAM recommended)
- **Camera:** Pi Camera Module 3 (libcamera/Picamera2) or any USB / V4L2 camera — `classification.py` tries Picamera2 first and falls back to OpenCV `VideoCapture`
- **Storage:** 32GB MicroSD Card  
  - Recommended: *SanDisk Ultra / A2 class* for better I/O stability

### Software & Network

- **OS:** Raspberry Pi OS (64-bit), flashed via **Raspberry Pi Imager**
- **Network:** Target Raspberry Pi must be on the **same WiFi / LAN** as your deployment PC

---

## 🚀 Stage 1: OS Provisioning & Access

1. Open **Raspberry Pi Imager** on your PC.
2. Select: **Raspberry Pi OS (64-bit)**.
3. Click the **⚙️ Settings** icon and configure:

   - ✅ Enable **SSH** (use *password authentication*)
   - ✅ Set **username** and **password**  
     - Example: `raspberry / <your_password>`
   - ✅ Configure **Wireless LAN**  
     - SSID: your WiFi name  
     - Password: your WiFi password

4. Write the OS to the SD card.
5. Insert the SD card into the Raspberry Pi 5 and power it on.
6. Wait about **2 minutes** for the initial boot to complete.

---

## 💻 Stage 2: Environment Setup & File Transfer

### 1. SSH into the Edge Device

From your **Computer terminal** (not on the Pi):

```bash
ssh raspberry@<PI5_IP>

Replace <PI5_IP> with the actual IP address of your Raspberry Pi 5.

2. Install System Dependencies (Docker & Python)

On the Raspberry Pi (over SSH):

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Python Edge-AI dependencies (Strictly Headless)
pip install opencv-python-headless onnxruntime numpy requests --break-system-packages

⚠️ Architect Note
We explicitly use opencv-python-headless instead of opencv-python to prevent X11 server dependencies and GUI crashes during background execution on a headless device.

3. Create Workspace & Transfer Project Files

On the Raspberry Pi:

mkdir -p ~/basil_project

You need at least:

classification.py

basil_mobilenet.onnx

requirements.txt (optional but recommended)

You can transfer your files using one of the methods below.

<details> <summary><b>Method A: SCP from PC (Recommended)</b></summary>

From your PC terminal:

scp "C:\path\to\classification.py" raspberry@<PI5_IP>:~/basil_project/
scp "C:\path\to\basil_mobilenet.onnx" raspberry@<PI5_IP>:~/basil_project/
</details> <details> <summary><b>Method B: Download via Google Drive (on the Pi)</b></summary>

On the Raspberry Pi:

pip install gdown --break-system-packages

# Ensure the Drive link is set to "Anyone with the link can view"
python3 -m gdown https://drive.google.com/uc?id=<FILE_ID> \
    -O ~/basil_project/basil_mobilenet.onnx

Replace <FILE_ID> with your actual Google Drive file ID.

</details>
⚙️ Stage 3: Headless Code Optimization

Since this system runs autonomously in a greenhouse, all GUI display calls must be disabled.

1. Disable OpenCV GUI Bindings

On the Raspberry Pi:

# Comment out cv2 display functions
sed -i 's/cv2.imshow/# cv2.imshow/' ~/basil_project/classification.py
sed -i 's/cv2.destroyAllWindows/# cv2.destroyAllWindows/' ~/basil_project/classification.py

Then manually edit the script:

nano ~/basil_project/classification.py

Locate the waitKey loop (or equivalent quit-condition logic) and replace:

# FIND:
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

with:

# REPLACE WITH:
pass  # GUI display disabled for headless deployment

Save and exit (Ctrl + O, Enter, Ctrl + X).

2. Dry Run Test

Run the script directly:

cd ~/basil_project && python3 classification.py

Expected console output example:

Ready. Classes: ['background', 'basil_healthy', 'disease']
Camera: Picamera2 (RGB format)   # or "Camera: USB (BGR format)" with a V4L2 device
disease | conf:0.72 | disease

If you see class predictions streaming without errors, you’re ready for production deployment.

🏭 Stage 4: Production Deployment (Auto-Start via systemd)

To ensure the system automatically recovers from power outages, we wrap the script in a systemd service.

1. Create the Service File

On the Raspberry Pi:

sudo bash -c 'cat > /etc/systemd/system/basil.service << EOF
[Unit]
Description=Basil Disease Edge Inference Service
After=network.target

[Service]
Type=simple
User=raspberry
WorkingDirectory=/home/raspberry/basil_project
ExecStart=/usr/bin/python3 /home/raspberry/basil_project/classification.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF'

🔁 Restart=always + RestartSec=5 ensures the pipeline automatically restarts if it crashes.

2. Enable and Start the Service
sudo systemctl daemon-reload
sudo systemctl enable basil.service
sudo systemctl start basil.service
3. Verify Status
sudo systemctl status basil.service

You should see something like:

● basil.service - Basil Disease Edge Inference Service
     Loaded: loaded (/etc/systemd/system/basil.service; enabled; vendor preset: enabled)
     Active: active (running) ✅
🩺 Maintenance & Troubleshooting
Operations Cheatsheet
Command	Action
journalctl -u basil.service -f	View live inference logs
sudo systemctl restart basil.service	Restart the AI pipeline
sudo systemctl stop basil.service	Stop the pipeline for debugging
Common Issues
1. ModuleNotFoundError: No module named 'cv2'

Cause: Standard OpenCV installed instead of headless.

Fix:

pip install opencv-python-headless --break-system-packages
2. ONNXRuntime Version Error

Cause: Version pinned or incompatible wheel for the device.

Fix:

pip install onnxruntime --break-system-packages

Avoid pinning specific ONNXRuntime versions on edge devices unless absolutely necessary.

3. Telegram Alerts Not Firing

The alerting logic requires 3–5 consecutive frames of disease detection before sending a notification.

Checklist:

Ensure the camera is steady and consistently pointing at the diseased leaf.

Verify that:

Model is loading correctly.

Class names are aligned with the alert logic.

Network connectivity for Telegram API is available.

💽 SD Card Architecture Strategy

To prevent dependency conflicts and maximize uptime in production, we follow a strict:

“One Card = One Project” isolation strategy.

🟩 SD Card 1 (New):

Project: Basil Classification

Service: basil.service active

🟨 SD Card 2 (Old):

Project: Durian Detection

Service: durian.service active

Operational rule:

Swapping SD cards = swapping entire projects.

This guarantees:

No virtual environment conflicts

No competing systemd services

Minimal operator error in the field

✅ Summary

By following this SOP, you get:

A fully headless, auto-starting basil disease classifier

Running on Raspberry Pi 5 with Picamera2 or a USB camera

Robust enough for real greenhouse deployment with near zero maintenance

Feel free to adapt the scripts and service definitions for other crops, models, or additional edge AI workloads.
