"""
Convert a trained PyTorch checkpoint (.pth) to ONNX format for deployment
on Raspberry Pi (ONNX Runtime inference).

Usage:
1. Set MODEL_NAME below to the model you want to convert.
2. Set CHECKPOINT_DIR to your actual checkpoint folder.
3. Run: python convert_to_onnx.py
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

# ============ Edit here ============
MODEL_NAME = "MobileNetV2"  # e.g. "MobileNetV3-Small"
CHECKPOINT_DIR = "/path/to/your/checkpoints"
OUTPUT_DIR = "/path/to/your/onnx_models"
NUM_CLASSES = 3  # REAL_ONLY setup: Background / Healthy / Disease
OPSET_VERSION = 19
# ====================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_model(model_name, num_classes):
    """Must match the model-building logic used during training, so the
    architecture lines up with the checkpoint's state_dict."""
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
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Files actually present in that folder:")
        if os.path.exists(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                print(f"   - {f}")
        return

    print(f"Building model: {MODEL_NAME} (num_classes={NUM_CLASSES})")
    model = create_model(MODEL_NAME, NUM_CLASSES)

    print(f"Loading weights: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Export to ONNX
    output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.onnx")
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting ONNX (opset={OPSET_VERSION}) -> {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET_VERSION,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"\nConversion done: {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX file size: {size_mb:.2f} MB")

    # Sanity-check that the exported model loads and runs
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": test_input})
        print(f"\nVerified: ONNX output shape = {outputs[0].shape}")
    except ImportError:
        print("\n(onnxruntime not installed, skipping verification. Try: pip install onnxruntime)")
    except Exception as e:
        print(f"\nVerification failed: {e}")


if __name__ == "__main__":
    main()
