!pip install --upgrade gdown -q

import gdown
import os
import re

DOWNLOAD_DIR = "/workspace/dataset"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Google Drive file URL -> target filename
# Replace these with your own Google Drive file links.
files = {
    "https://drive.google.com/file/d/YOUR_FILE_ID_1/view?usp=drive_link": "Real_disease.zip",
    "https://drive.google.com/file/d/YOUR_FILE_ID_2/view?usp=drive_link": "Proxy_disease.zip",
    "https://drive.google.com/file/d/YOUR_FILE_ID_3/view?usp=drive_link": "Basil_healthy.zip",
    "https://drive.google.com/file/d/YOUR_FILE_ID_4/view?usp=drive_link": "Background.zip",
}

def extract_file_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

for url, filename in files.items():
    file_id = extract_file_id(url)
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    print(f"Downloading {filename} (id={file_id}) ...")
    # Plain uc?id= format works across all gdown versions, no fuzzy flag needed
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

print("\n========== Download check ==========")
for filename in files.values():
    path = os.path.join(DOWNLOAD_DIR, filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{filename}: {size_mb:.2f} MB")
    else:
        print(f"{filename}: download failed, file not found")
