import zipfile
import os
import shutil

# ============ 改这里 ============
# zip文件直接下载到了 /workspace/dataset，跟解压目标用同一个目录也没问题
UPLOAD_DIR = "/workspace/dataset"
LOCAL_DATASET = "/workspace/dataset"
# =================================

os.makedirs(LOCAL_DATASET, exist_ok=True)

zip_files = {
    "Background.zip": "Background",
    "Basil_healthy.zip": "Basil_healthy",
    "Real_disease.zip": "Real_disease",
    "Proxy_disease.zip": "Proxy_disease",
}

def extract_and_flatten(zip_path, target_folder):
    # 解压zip到target_folder，如果zip内部多包了一层同名文件夹就拍平
    tmp_extract = target_folder + "_tmp_extract"
    os.makedirs(tmp_extract, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_extract)

    entries = os.listdir(tmp_extract)

    if len(entries) == 1 and os.path.isdir(os.path.join(tmp_extract, entries[0])):
        inner = os.path.join(tmp_extract, entries[0])
        os.makedirs(target_folder, exist_ok=True)
        for item in os.listdir(inner):
            shutil.move(os.path.join(inner, item), os.path.join(target_folder, item))
    else:
        os.makedirs(target_folder, exist_ok=True)
        for item in entries:
            shutil.move(os.path.join(tmp_extract, item), os.path.join(target_folder, item))

    shutil.rmtree(tmp_extract, ignore_errors=True)


for zip_name, folder_name in zip_files.items():
    zip_path = os.path.join(UPLOAD_DIR, zip_name)
    target_folder = os.path.join(LOCAL_DATASET, folder_name)

    if not os.path.exists(zip_path):
        print(f"WARNING: 找不到 {zip_path}，检查一下上传路径/文件名对不对")
        continue

    print(f"解压 {zip_name} -> {target_folder} ...")
    extract_and_flatten(zip_path, target_folder)
    print("完成")

print("\n========== 解压结果检查 ==========")
for folder_name in zip_files.values():
    folder_path = os.path.join(LOCAL_DATASET, folder_name)
    if os.path.exists(folder_path):
        count = sum(
            1 for root, _, files in os.walk(folder_path)
            for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        print(f"{folder_name}: {count} 张图片 (含子文件夹)")
    else:
        print(f"{folder_name}: 文件夹不存在")
