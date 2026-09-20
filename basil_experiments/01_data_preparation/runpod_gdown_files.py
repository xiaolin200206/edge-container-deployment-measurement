!pip install --upgrade gdown -q

import gdown
import os
import re

DOWNLOAD_DIR = "/workspace/dataset"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 文件链接 -> 目标文件名
files = {
    "https://drive.google.com/file/d/167RZr07BEGNJgBTykSKGinfXnqogATr2/view?usp=drive_link": "Real_disease.zip",
    "https://drive.google.com/file/d/1OODQbLfQF3jVs0jqh_RhZ0fzZTN-xu9S/view?usp=drive_link": "Proxy_disease.zip",
    "https://drive.google.com/file/d/1URbtwaAHznWg16lrDHiznoUd3yIOJtaI/view?usp=drive_link": "Basil_healthy.zip",
    "https://drive.google.com/file/d/16WOeK881HA48tYWS4oMiR9H3tmM2YZzs/view?usp=drive_link": "Background.zip",
}

def extract_file_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

for url, filename in files.items():
    file_id = extract_file_id(url)
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    print(f"下载 {filename} (id={file_id}) ...")
    # 用经典的uc?id=格式，兼容所有gdown版本，不需要fuzzy参数
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

print("\n========== 下载结果检查 ==========")
for filename in files.values():
    path = os.path.join(DOWNLOAD_DIR, filename)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{filename}: {size_mb:.2f} MB")
    else:
        print(f"{filename}: 下载失败，文件不存在")
