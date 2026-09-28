"""
把训练好的PyTorch checkpoint (.pth) 转换成ONNX格式
用于部署到Raspberry Pi (ONNX Runtime推理)

使用方法：
1. 改下面的 MODEL_NAME，选择要转换的模型
2. 改 CHECKPOINT_DIR 确认路径正确
3. 运行：python convert_to_onnx.py
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

# ============ 改这里 ============
MODEL_NAME = "MobileNetV2"  # 改成你要转换的模型名，比如 "MobileNetV3-Small"
CHECKPOINT_DIR = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\Third paper\real only result\checkpoints"
OUTPUT_DIR = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\Third paper\onnx_models"
NUM_CLASSES = 3  # REAL_ONLY配置: Background / Healthy / Disease
OPSET_VERSION = 19
# =================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_model(model_name, num_classes):
    """跟训练脚本里完全一样的模型创建逻辑，确保架构匹配checkpoint"""
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    elif model_name == "MobileNetV3-Small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1024, num_classes))

    elif model_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1280, num_classes))

    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    elif model_name == "EfficientNet-B1":
        model = models.efficientnet_b1(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    elif model_name == "ShuffleNetV2-0.5x":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "SqueezeNet":
        model = models.squeezenet1_1(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        original_forward = model.forward
        def new_forward(x):
            x = original_forward(x)
            return x.view(x.size(0), -1)
        model.forward = new_forward

    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)

    elif model_name == "DenseNet-121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, num_classes)

    elif model_name == "RegNet-Y400MF":
        model = models.regnet_y_400mf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "ViT-Tiny":
        from torchvision.models import vit_b_16
        model = vit_b_16(weights=None)
        model.heads = nn.Linear(model.heads[0].in_features, num_classes)

    else:
        raise ValueError(f"未知模型: {model_name}")

    return model


def main():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"找不到checkpoint: {checkpoint_path}")
        print(f"该文件夹下实际有的文件:")
        if os.path.exists(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                print(f"   - {f}")
        return

    print(f"加载模型架构: {MODEL_NAME} (num_classes={NUM_CLASSES})")
    model = create_model(MODEL_NAME, NUM_CLASSES)

    print(f"加载权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 导出ONNX
    output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.onnx")
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"导出ONNX (opset={OPSET_VERSION}) -> {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET_VERSION,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"\n转换完成: {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX文件大小: {size_mb:.2f} MB")

    # 验证一下ONNX模型能不能正常加载+推理
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": test_input})
        print(f"\n验证通过: ONNX推理输出shape = {outputs[0].shape}")
    except ImportError:
        print("\n(未安装onnxruntime，跳过验证。建议: pip install onnxruntime)")
    except Exception as e:
        print(f"\n验证失败: {e}")


if __name__ == "__main__":
    main()
