# MRI Image Preprocessing Pipeline

用于多序列腹部 MRI 数据预处理的 Python 工具集，覆盖 DICOM 转 NIfTI、序列配准、裁剪与强度归一化，以及低分辨率/高分辨率样本生成。

## 功能

- 自动识别并转换常见 MRI 序列：`T2FS`、`DWI`、`ADC`、`DixonW`、`DixonF`、`DixonIP`、`DixonOP`、`DynT1W`
- 将多序列图像配准到 `T2FS` 空间
- 基于前景掩膜进行 ROI 裁剪、强度截断、z-score 归一化和 XY 尺寸调整
- 生成 `256 x 256` 或 `512 x 512` 的 HR/LR 数据
- 输出处理元数据、质量控制指标、CSV/XLSX 报告和可视化结果
- 支持断点续跑：已生成的病例/文件会自动跳过

## 项目结构

```text
.
├── preprocess/
│   ├── convert.py               # DICOM -> NIfTI 与序列识别
│   ├── register.py              # 配准到 T2FS
│   ├── normalize_256.py         # 裁剪、归一化、调整到 256 x 256
│   ├── normalize_512.py         # 裁剪、归一化、调整到 512 x 512
│   ├── generate_lr_hr_256.py    # 生成 256 尺寸 LR/HR 样本
│   ├── generate_lr_hr_512.py    # 生成 512 尺寸 LR/HR 样本
│   └── pipeline.py              # 命令行流水线入口
├── tests/
├── config.example.env
├── requirements.txt
├── LICENSE
└── README.md
```

## 环境要求

- Python 3.10+
- `dcm2niix`，并确保可通过 `dcm2niix` 命令调用，或在配置中填写其绝对路径
- 能够读取 DICOM 和 NIfTI 的运行环境

安装 Python 依赖：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 数据目录约定

原始 DICOM 目录按病例组织：

```text
data/raw_dicom/
└── Center1-000001/
    ├── 1_t2_fblade_fs_tra/
    ├── 2_ep2d_diff_b50_800_ADC/
    └── ...
```

流水线会依次产生 `data/nii`、`data/registered_to_T2FS`、`data/cropped_normalized_256`、`data/cropped_normalized_512`、`data/lr_hr_256` 和 `data/lr_hr_512`。

请勿将真实患者数据、DICOM 文件、NIfTI 文件、日志或生成结果提交到 GitHub。仓库中的 `.gitignore` 已默认忽略这些目录。

## 快速开始

复制配置模板并修改路径：

```bash
cp config.example.env config.local.env
```

Windows PowerShell：

```powershell
Copy-Item config.example.env config.local.env
```

Linux/macOS 载入配置：

```bash
set -a
source config.local.env
set +a
```

运行完整流程：

```bash
python -m preprocess.pipeline
```

只运行指定阶段：

```bash
python -m preprocess.pipeline --stages convert register normalize256 generate256
```

可用阶段：`convert`、`register`、`normalize256`、`normalize512`、`generate256`、`generate512`。

## 单独运行脚本

```bash
python -m preprocess.convert
python -m preprocess.register
python -m preprocess.normalize_256
python -m preprocess.normalize_512
python -m preprocess.generate_lr_hr_256
python -m preprocess.generate_lr_hr_512
```

## 输出说明

- 转换阶段：标准化命名的 `.nii.gz`、可选 JSON 元数据和 `metadata.json`
- 配准阶段：`*_to_T2FS.nii.gz` 和 `registration_report.xlsx`
- 归一化阶段：`{sequence}_crop.nii.gz`、`qc.json` 以及汇总 CSV
- LR/HR 阶段：`HR_T2FS.nii.gz`、不同缩放倍数的 LR 文件、可视化 PNG 和生成日志

## 数据与伦理

本项目只提供处理代码，不包含任何临床数据。使用真实患者数据前，请完成去标识化、伦理审批、访问控制和本地数据安全评估。不同扫描仪、序列协议和数据集可能需要重新校准序列识别规则及归一化参数。


## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
