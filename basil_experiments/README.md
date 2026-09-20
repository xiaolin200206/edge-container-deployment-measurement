# Experiment Scripts — Greenhouse Basil Disease Classification

Scripts referenced in the manuscript "Containerized Edge Intelligence for
Greenhouse Disease Monitoring: A Lightweight Classification System with
Real-Time Alerting on Raspberry Pi 5" (under review). Each script is provided in two language versions with identical
logic — `en/` for English comments, `zh/` for Chinese comments.

中文说明见下方对应小节。

---

## Folder Structure

```
basil_experiments/
├── 01_data_preparation/   {en,zh}/runpod_gdown_files.py, runpod_extract.py
├── 02_baseline_comparison/{en,zh}/runpod_train_real_only.py
│                                  runpod_train_real_plus_proxy.py
│                                  runpod_train_real_plus_proxy_balanced.py
├── 03_analysis/           {en,zh}/threshold_sensitivity_analysis.py
│                                  ood_proxy_analysis.py
│                                  system_analysis.py
└── 04_deployment/         {en,zh}/convert_to_onnx.py
                                   basil_classifier.py
```

## What You Need Before Starting

- A dataset organized as four folders: `Background/`, `Basil_healthy/`,
  `Real_disease/`, `Proxy_disease/` (each containing .jpg/.jpeg/.png images).
  This dataset is **not included** in this repository — see the manuscript's
  Data Availability Statement.
- A cloud GPU instance (the scripts were run on RunPod; any CUDA-capable
  instance with Jupyter/notebook access works) for steps 1–3.
- A Raspberry Pi 5 with Camera Module 3 for step 4.
- Python 3.9+, PyTorch, torchvision, scikit-learn, pandas, matplotlib.

---

## Step-by-Step Reproduction

### Step 1 — Get the dataset onto your GPU instance (`01_data_preparation/`)

`runpod_gdown_files.py` and `runpod_extract.py` are written as **Jupyter
notebook cells** (note the `!pip install` line at the top of the gdown
script) — paste each into its own notebook cell rather than running them as
standalone `.py` files.

1. Upload your four dataset zip files to Google Drive and replace the
   placeholder `YOUR_FILE_ID_*` links in `runpod_gdown_files.py` with your
   own Drive file links.
2. Run `runpod_gdown_files.py` in a notebook cell — downloads the four zips
   into `/workspace/dataset`.
3. Run `runpod_extract.py` in the next cell — extracts and flattens each
   zip into `/workspace/dataset/{Background,Basil_healthy,Real_disease,Proxy_disease}/`.
4. Confirm the printed image counts per folder look right before continuing.

### Step 2 — Train and compare 11 architectures (`02_baseline_comparison/`)

Maps to **Section 4.1 / Table 2** of the manuscript.

Run each of the three scripts in turn (each is a standalone script, ~15–30 min
per architecture on a single GPU, ~11 architectures per script):

```bash
python runpod_train_real_only.py
python runpod_train_real_plus_proxy.py
python runpod_train_real_plus_proxy_balanced.py
```

- `runpod_train_real_only.py` — Disease class uses only `Real_disease`.
- `runpod_train_real_plus_proxy.py` — Disease class is the unbalanced union
  of `Real_disease` + `Proxy_disease` (sample size confound; kept for
  transparency, superseded by the balanced run below).
- `runpod_train_real_plus_proxy_balanced.py` — Disease class is resampled to
  match `Real_disease`'s count exactly (280 real + 280 proxy, seed=42). **This
  is the sample-size-controlled comparison reported in Table 2.**

Each script saves a checkpoint after every architecture finishes (so a
dropped session doesn't lose completed results) and writes a final summary
CSV, e.g. `baseline_comparison_REAL_ONLY_final.csv`.

### Step 3 — Run the analyses (`03_analysis/`)

These read the CSV logs produced by a field deployment (see Step 4) — you
need a `basil_data.csv` inference log (and `cycle_events.csv` for the system
analysis script) before running these. Update the `CSV_FILE` / `OUTPUT_DIR`
constants at the top of each script to point at your own data/output folders.

```bash
python threshold_sensitivity_analysis.py   # Section 4.2.1 / Table 3
python ood_proxy_analysis.py               # Section 5.2.1
python system_analysis.py             # Table 1 / Figure 6
```

- `threshold_sensitivity_analysis.py` — Replays the field log under a 3×3
  grid of confidence thresholds (0.3/0.5/0.7) × temporal debouncing window
  sizes (3/5/7 frames). Outputs `threshold_sensitivity_table.csv`.
- `ood_proxy_analysis.py` — Confidence-based proxy analysis: low-confidence
  frame clustering, class-flip rate comparison, correlation against system
  telemetry. Outputs `ood_proxy_analysis_charts.png` + summary text.
- `system_analysis.py` — Aggregates latency/CPU/RAM/temperature
  statistics into the paper's hardware summary table and charts.

### Step 4 — Export and deploy to Raspberry Pi 5 (`04_deployment/`)

1. **Export the trained model to ONNX** (run on the machine where your
   `.pth` checkpoint lives, not on the Pi):
   ```bash
   python convert_to_onnx.py
   ```
   Edit `MODEL_NAME`, `CHECKPOINT_DIR`, and `OUTPUT_DIR` at the top of the
   script first.

2. **Copy the resulting `.onnx` file to the Raspberry Pi 5**, into the same
   folder as `basil_classifier.py`.

3. **Set your Telegram credentials as environment variables** (never edit
   them directly into the script):
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
   ```
   If you skip this, the script still runs normally — it just won't send alerts.

4. **Run the deployment script** on the Pi:
   ```bash
   python3 basil_classifier.py
   ```
   This captures frames via Picamera2, runs ONNX Runtime inference, applies
   temporal debouncing (manuscript Eq. 2), logs telemetry to CSV, and sends
   a Telegram alert with an annotated image when a Disease state is confirmed.

   To also see the live preview window, run with `DISPLAY=:0` so it renders
   on the Pi's own HDMI output (a plain SSH session has no display to show it on):
   ```bash
   DISPLAY=:0 python3 basil_classifier.py
   ```

---

## Reproducibility Notes
- Random seed fixed at 42 across all training and sampling steps.
- All 11 architectures trained under identical hyperparameters (Adam
  optimizer, batch size 32, ImageNet-pretrained initialization, 15 epochs).
- The image dataset itself is not included — see the manuscript's Data
  Availability Statement for access conditions.

---

## 中文说明

### 目录结构同上，每个脚本都有`en/`（英文注释）和`zh/`（中文注释）两个版本，逻辑完全一致。

### 准备工作
- 数据集要整理成4个文件夹：`Background/`、`Basil_healthy/`、`Real_disease/`、`Proxy_disease/`，每个里面放jpg/jpeg/png图片。**数据集本身不在这个仓库里**，获取方式见论文里的Data Availability Statement。
- 需要一台能跑CUDA的云GPU（原始实验在RunPod上跑的，任何支持Jupyter notebook的GPU实例都行），用于第1-3步。
- 需要一台Raspberry Pi 5 + Camera Module 3，用于第4步。

### 第1步：把数据集传到GPU实例上（`01_data_preparation/`）

`runpod_gdown_files.py`和`runpod_extract.py`是写给**Jupyter notebook cell**用的（注意gdown那个脚本开头有个`!pip install`，这是notebook专属语法），**不要直接用`python3 xxx.py`跑**，要分别粘贴到notebook的两个cell里执行。

1. 把你自己的4个数据集zip传到Google Drive，把`runpod_gdown_files.py`里的`YOUR_FILE_ID_*`占位符换成你自己的Drive链接。
2. 在notebook里跑`runpod_gdown_files.py`这个cell——下载4个zip到`/workspace/dataset`。
3. 下一个cell跑`runpod_extract.py`——解压并拍平到`/workspace/dataset/{Background,Basil_healthy,Real_disease,Proxy_disease}/`。
4. 确认打印出来的每个文件夹图片数量正常，再往下走。

### 第2步：训练+对比11个架构（`02_baseline_comparison/`）

对应论文**4.1节/Table 2**。

依次跑这三个脚本（每个脚本单卡大概15-30分钟/架构，11个架构）：

```bash
python runpod_train_real_only.py
python runpod_train_real_plus_proxy.py
python runpod_train_real_plus_proxy_balanced.py
```

- `runpod_train_real_only.py`——Disease类别只用`Real_disease`。
- `runpod_train_real_plus_proxy.py`——Disease类别是`Real_disease`+`Proxy_disease`不受控的合并（存在数据量混杂，留着是为了透明，已被下面balanced版本取代）。
- `runpod_train_real_plus_proxy_balanced.py`——Disease类别重新抽样成跟`Real_disease`数量完全一致（280真实+280代理，seed=42）。**这才是Table 2里报告的、控制了样本量的对比**。

每个脚本跑完一个架构就立刻存一次checkpoint（这样断线也不会丢已完成的结果），最后输出一个汇总CSV，比如`baseline_comparison_REAL_ONLY_final.csv`。

### 第3步：跑分析脚本（`03_analysis/`）

这几个脚本要读取现场部署产生的CSV日志（见第4步）——跑之前你需要先有一份`basil_data.csv`推理日志（跑系统分析脚本还需要`cycle_events.csv`）。先把每个脚本开头的`CSV_FILE`/`OUTPUT_DIR`改成你自己的路径。

```bash
python threshold_sensitivity_analysis.py   # 对应4.2.1节/Table 3
python ood_proxy_analysis.py               # 对应5.2.1节
python system_analysis.py             # 对应Table 1/Figure 6
```

- `threshold_sensitivity_analysis.py`——把现场日志在3×3组合（阈值0.3/0.5/0.7 × 时序窗口3/5/7帧）下重新回放一遍。输出`threshold_sensitivity_table.csv`。
- `ood_proxy_analysis.py`——基于置信度的代理分析：低置信度帧聚集性、类别震荡率对比、跟系统遥测数据的相关性。输出图表+文字摘要。
- `system_analysis.py`——把延迟/CPU/RAM/温度数据汇总成论文里的硬件summary表和图。

### 第4步：导出ONNX并部署到Raspberry Pi 5（`04_deployment/`）

1. **把训练好的模型导出成ONNX**（在你存checkpoint的那台机器上跑，不是在Pi上跑）：
   ```bash
   python convert_to_onnx.py
   ```
   先把脚本开头的`MODEL_NAME`、`CHECKPOINT_DIR`、`OUTPUT_DIR`改成你自己的。

2. **把导出的`.onnx`文件传到Raspberry Pi 5**，跟`basil_classifier.py`放在同一个文件夹。

3. **把Telegram凭证设成环境变量**（不要直接改在脚本里）：
   ```bash
   export TELEGRAM_BOT_TOKEN="你的bot token"
   export TELEGRAM_CHAT_ID="你的chat id"
   ```
   不设置的话脚本照样能跑，只是不会发送告警。

4. **在Pi上跑部署脚本**：
   ```bash
   python3 basil_classifier.py
   ```
   会用Picamera2拍照、跑ONNX Runtime推理、做时序去抖（对应论文公式2）、把遥测数据记录到CSV，确认是Disease状态时发Telegram告警（带标注过的图片）。

   如果还想看实时画面，要加`DISPLAY=:0`，这样画面才会显示在Pi自己接的HDMI屏幕上（纯SSH窗口没有图形界面，显示不出来）：
   ```bash
   DISPLAY=:0 python3 basil_classifier.py
   ```

### 复现性说明
- 所有训练/抽样步骤固定seed=42。
- 11个架构全部用相同超参数训练（Adam优化器，batch size 32，ImageNet预训练权重初始化，15个epoch）。
- 数据集本身不包含在这个仓库里，获取方式见论文的Data Availability Statement。
