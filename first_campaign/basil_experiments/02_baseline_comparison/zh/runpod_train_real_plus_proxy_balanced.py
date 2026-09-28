# CELL 2 — 11模型 REAL_ONLY 训练对比（CELL 1跑完确认数量对了，再跑这个）
# ============================================================

import os
import time
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
DATA_PATH = "/workspace/dataset"  # runpod_extract.py 解压到的路径
DATASET_MODE = "REAL_PLUS_PROXY_BALANCED"
# 结果直接存回Drive，session断了/睡醒了都还在
OUTPUT_PATH = f"/workspace/baseline_results_{DATASET_MODE.lower()}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4  # RunPod环境，vCPU较多可以适当调高
NUM_EPOCHS = 15
# ==========================================

print(f"设备: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
print(f"数据路径: {DATA_PATH}")
print(f"输出路径 (Drive): {OUTPUT_PATH}")
print(f"配置: mode={DATASET_MODE}, batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}\n")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'), exist_ok=True)


# ============ 数据加载 ============
class LeafDataset(Dataset):
    """
    支持3种配置:
      REAL_ONLY              : Disease = 全部Real_disease，不用Proxy
      REAL_PLUS_PROXY        : Disease = Real_disease + Proxy_disease 全部合并（数据量会变多，不受控）
      REAL_PLUS_PROXY_BALANCED: Disease总数控制成跟Real_disease原始数量一致，
                                内部50/50随机抽样混合real+proxy（固定seed可复现），
                                用于剔除"数据量变多"这个混杂变量，干净对比domain shift。
    """
    SKIP_DIRS = {'.ipynb_checkpoints', '__pycache__'}
    IMG_EXTS = ('.jpg', '.jpeg', '.png')
    BALANCE_SEED = 42  # 固定seed，保证抽样可复现

    def __init__(self, root_dir, transform=None, mode="REAL_ONLY"):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = {}
        self.class_idx = 0

        print(f"数据集配置: {mode}")

        def collect_images_recursive(folder):
            found = []
            for entry in sorted(os.listdir(folder)):
                entry_path = os.path.join(folder, entry)
                if os.path.isdir(entry_path):
                    if entry in self.SKIP_DIRS:
                        continue
                    found.extend(collect_images_recursive(entry_path))
                elif entry.lower().endswith(self.IMG_EXTS):
                    found.append(entry_path)
            return found

        background_path = os.path.join(root_dir, 'Background')
        healthy_path = os.path.join(root_dir, 'Basil_healthy')
        real_path = os.path.join(root_dir, 'Real_disease')
        proxy_path = os.path.join(root_dir, 'Proxy_disease')

        background_imgs = collect_images_recursive(background_path) if os.path.isdir(background_path) else []
        healthy_imgs = collect_images_recursive(healthy_path) if os.path.isdir(healthy_path) else []
        real_imgs = collect_images_recursive(real_path) if os.path.isdir(real_path) else []
        proxy_imgs = collect_images_recursive(proxy_path) if os.path.isdir(proxy_path) else []

        print(f"   Background 原始: {len(background_imgs)}张")
        print(f"   Basil_healthy 原始: {len(healthy_imgs)}张")
        print(f"   Real_disease 原始: {len(real_imgs)}张")
        print(f"   Proxy_disease 原始: {len(proxy_imgs)}张")

        if mode == "REAL_ONLY":
            disease_imgs = real_imgs

        elif mode == "REAL_PLUS_PROXY":
            disease_imgs = real_imgs + proxy_imgs

        elif mode == "REAL_PLUS_PROXY_BALANCED":
            target_n = len(real_imgs)  # Disease类别总数控制成跟Real Only一致
            half = target_n // 2
            other_half = target_n - half

            rng = random.Random(self.BALANCE_SEED)
            real_sample = rng.sample(real_imgs, min(half, len(real_imgs)))
            proxy_sample = rng.sample(proxy_imgs, min(other_half, len(proxy_imgs)))

            disease_imgs = real_sample + proxy_sample
            print(f"   [BALANCED抽样] 目标总数={target_n} -> real抽{len(real_sample)}张 + proxy抽{len(proxy_sample)}张"
                  f" = {len(disease_imgs)}张 (seed={self.BALANCE_SEED})")
        else:
            raise ValueError(f"未知mode: {mode}")

        images_by_class = {
            'Background': background_imgs,
            'Healthy': healthy_imgs,
            'Disease': disease_imgs,
        }

        for class_label in ['Background', 'Healthy', 'Disease']:
            imgs = images_by_class[class_label]
            if len(imgs) == 0:
                print(f"跳过空类别: {class_label}")
                continue
            self.class_names[self.class_idx] = class_label
            for img_path in imgs:
                self.images.append(img_path)
                self.labels.append(self.class_idx)
            self.class_idx += 1

        print(f"加载数据集 ({mode}): 共{len(self.images)}张图片，{self.class_idx}个类别")
        for idx, name in self.class_names.items():
            count = sum(1 for l in self.labels if l == idx)
            print(f"   - {name}: {count}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            if self.transform:
                img = torch.zeros(3, 224, 224)
            return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n加载数据集...")
dataset = LeafDataset(DATA_PATH, transform=transform, mode=DATASET_MODE)
num_classes = dataset.class_idx

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}\n")


# ============ 模型创建 ============
def create_model(model_name, num_classes):
    try:
        if model_name == "MobileNetV2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

        elif model_name == "MobileNetV3-Small":
            model = models.mobilenet_v3_small(pretrained=True)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 1024), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

        elif model_name == "MobileNetV3-Large":
            model = models.mobilenet_v3_large(pretrained=True)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 1280), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1280, num_classes))

        elif model_name == "EfficientNet-B0":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

        elif model_name == "EfficientNet-B1":
            model = models.efficientnet_b1(pretrained=True)
            model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

        elif model_name == "ShuffleNetV2-0.5x":
            model = models.shufflenet_v2_x0_5(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "SqueezeNet":
            model = models.squeezenet1_1(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
            original_forward = model.forward
            def new_forward(x):
                x = original_forward(x)
                return x.view(x.size(0), -1)
            model.forward = new_forward

        elif model_name == "ResNet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, num_classes)

        elif model_name == "DenseNet-121":
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(1024, num_classes)

        elif model_name == "RegNet-Y400MF":
            model = models.regnet_y_400mf(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "ViT-Tiny":
            from torchvision.models import vit_b_16
            model = vit_b_16(pretrained=True)
            model.heads = nn.Linear(model.heads[0].in_features, num_classes)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model
    except Exception as e:
        print(f"模型 {model_name} 创建失败: {e}")
        return None


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), accuracy, all_preds, all_labels


def benchmark_model(model, device, num_iterations=100):
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_latency = (elapsed / num_iterations) * 1000
    params = sum(p.numel() for p in model.parameters()) / 1e6
    param_size = sum(p.numel() * 4 / 1e6 for p in model.parameters())
    return avg_latency, params, param_size


# ============ 主流程 ============
models_to_test = [
    "MobileNetV2", "MobileNetV3-Small", "MobileNetV3-Large",
    "EfficientNet-B0", "EfficientNet-B1", "ShuffleNetV2-0.5x",
    "SqueezeNet", "ResNet18", "DenseNet-121", "RegNet-Y400MF", "ViT-Tiny"
]

results = []
results_csv_path = os.path.join(OUTPUT_PATH, 'baseline_comparison_progress.csv')

for i, model_name in enumerate(models_to_test, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(models_to_test)}] 训练模型: {model_name}")
    print(f"{'='*70}")

    model = create_model(model_name, num_classes)
    if model is None:
        print(f"跳过 {model_name}")
        continue
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc, best_epoch = 0, 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"  最佳准确率: {best_acc:.4f} (epoch {best_epoch})")

    _, final_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, DEVICE)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_latency, params, param_size = benchmark_model(model, DEVICE)

    print(f"  准确率: {final_acc:.4f} | 精确率: {precision:.4f} | 召回率: {recall:.4f} | F1: {f1:.4f}")
    print(f"  延迟: {avg_latency:.2f}ms | 参数: {params:.2f}M | 大小: {param_size:.2f}MB")

    results.append({
        'Model': model_name, 'Accuracy': round(final_acc, 4),
        'Precision': round(precision, 4), 'Recall': round(recall, 4), 'F1-Score': round(f1, 4),
        'Latency (ms)': round(avg_latency, 2), 'Parameters (M)': round(params, 2),
        'Model Size (MB)': round(param_size, 2),
    })

    # 每跑完一个模型就立刻存一次checkpoint + 更新CSV到Drive，
    # 这样即使Colab中途断线，已经跑完的模型结果不会丢
    checkpoint_path = os.path.join(OUTPUT_PATH, 'checkpoints', f'{model_name}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    pd.DataFrame(results).to_csv(results_csv_path, index=False)
    print(f"  已存到Drive: {checkpoint_path}")

    del model
    torch.cuda.empty_cache()

# ============ 最终汇总 ============
results_df = pd.DataFrame(results)
final_csv = os.path.join(OUTPUT_PATH, 'baseline_comparison_REAL_PLUS_PROXY_BALANCED_final.csv')
results_df.to_csv(final_csv, index=False)

print(f"\n{'='*70}")
print(f"全部完成！结果已保存到 Drive: {final_csv}")
print(f"{'='*70}")
print(results_df.sort_values('Accuracy', ascending=False).to_string(index=False))
