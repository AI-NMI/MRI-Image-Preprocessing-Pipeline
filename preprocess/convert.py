#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil
import tempfile
import subprocess
import logging
import nibabel as nib

from pathlib import Path
from datetime import datetime

# =========================================================
# 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# 日志
# =========================================================

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "convert.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

FAILED_CASES_FILE = LOG_DIR / "failed_cases.txt"

# =========================================================
# 序列列表
# =========================================================

ALL_SEQUENCES = [
    "T2FS",
    "DixonW",
    "DixonF",
    "DixonIP",
    "DixonOP",
    "DWI",
    "ADC",
    "DynT1W"
]

# =========================================================
# 排除关键词
# =========================================================

EXCLUDED_POS = [
    "survey",
    "localizer",
    "scout",
    "mip",
    "mrcp",
    "rad",
    "vsmrcp",
    "reformatted",
    "subtraction",
    "volume",
    "mask"
]

# =========================================================
# 工具函数
# =========================================================

def contains_any(name, keywords):

    return any(k in name for k in keywords)

# =========================================================
# 序列识别
# =========================================================

def identify_sequence(folder_name):

    name = folder_name.lower()

    # =====================================================
    # 排除
    # =====================================================

    if contains_any(name, EXCLUDED_POS):
        return None

    # =====================================================
    # ADC（必须在 DWI 前）
    # =====================================================

    if "adc" in name:
        return "ADC"

    # =====================================================
    # DWI
    # =====================================================

    dwi_keywords = [
        "dwi",
        "diff",
        "tracew",
        "b50",
        "b800",
        "b1000"
    ]

    if any(x in name for x in dwi_keywords):

        if "adc" not in name:
            return "DWI"

    # =====================================================
    # Dixon Water
    # =====================================================

    if (
        "water" in name
        or name.endswith("_w")
        or "_w_" in name
        or "_w." in name
        or " wo " in name
        or "_wo" in name
    ):
        return "DixonW"

    # =====================================================
    # Dixon Fat
    # =====================================================

    if (
        "fat" in name
        or name.endswith("_f")
        or "_f_" in name
        or "_f." in name
        or " wf " in name
        or "_wf" in name
    ):
        return "DixonF"

    # =====================================================
    # Dixon In Phase
    # =====================================================

    if (
        "inphase" in name
        or "_ip" in name
        or "_in" in name
        or " in " in name
        or name.endswith("in")
    ):
        return "DixonIP"

    # =====================================================
    # Dixon Opposed Phase
    # =====================================================

    if (
        "opposed" in name
        or "_op" in name
        or "_opp" in name
        or "_out" in name
        or " out " in name
        or name.endswith("out")
    ):
        return "DixonOP"

    # =====================================================
    # T2FS
    # =====================================================

    if (
        "t2" in name
        and (
            "fs" in name
            or "spair" in name
            or "spir" in name
            or "fat sat" in name
            or "fatsat" in name
            or "stir" in name
            or "tirm" in name
            or "prop" in name
            or "blade" in name
            or "fse" in name
            or "ssfse" in name
        )
    ):
        return "T2FS"

    # =====================================================
    # DynT1W
    # =====================================================

    is_t1_dynamic = any([
        "t1" in name,
        "lava" in name,
        "vibe" in name,
        "mdixon" in name
    ])

    has_dynamic_phase = any([
        "arterial" in name,
        "artery" in name,
        "portal" in name,
        "venous" in name,
        "delay" in name,
        "delayed" in name,
        "dynamic" in name,
        "dyn" in name,
        "+c" in name,
        " post" in name,
        " 2a " in name,
        " ap " in name,
        " vp " in name,
        "_bh" in name
    ])

    if is_t1_dynamic and has_dynamic_phase:
        return "DynT1W"

    return None

# =========================================================
# NIfTI 信息
# =========================================================

def get_nii_info(nii_path):

    try:

        img = nib.load(str(nii_path))

        shape = [int(x) for x in img.shape]

        spacing = [float(x) for x in img.header.get_zooms()]

        return {
            "shape": shape,
            "spacing": spacing
        }

    except Exception as e:

        logger.error(f"读取失败: {nii_path}")
        logger.error(str(e))

        return {
            "shape": None,
            "spacing": None
        }

# =========================================================
# DICOM -> NIfTI
# =========================================================

def convert_dicom_to_nifti(dicom_dir, output_path, seq_name):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        tmpdir = Path(tmpdir)

        cmd = [
            os.environ.get("DCM2NIIX_BIN", "dcm2niix"),
            "-z", "y",
            "-o", str(tmpdir),
            "-f", "%p_%s",
            str(dicom_dir)
        ]

        try:

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            nii_files = sorted(tmpdir.glob("*.nii.gz"))

            if len(nii_files) == 0:

                logger.error(f"[NO OUTPUT] {seq_name}")

                return False

            # =================================================
            # 默认取最大文件
            # =================================================

            nii_file = max(
                nii_files,
                key=lambda x: x.stat().st_size
            )

            shutil.move(
                str(nii_file),
                str(output_path)
            )

            # =================================================
            # JSON
            # =================================================

            json_files = sorted(tmpdir.glob("*.json"))

            if len(json_files) > 0:

                json_out = output_path.with_suffix("").with_suffix(".json")

                shutil.move(
                    str(json_files[0]),
                    str(json_out)
                )

            logger.info(f"转换成功: {seq_name}")

            return True

        except subprocess.CalledProcessError as e:

            logger.error(f"失败: {seq_name}")
            logger.error(e.stderr)

            return False

# =========================================================
# 单病例处理
# =========================================================

def process_one_center(center_id, raw_base, nii_base):

    raw_dir = raw_base / center_id

    if not raw_dir.exists():

        return {
            "center_id": center_id,
            "converted": [],
            "failed": []
        }

    seq_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

    out_dir = nii_base / center_id
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    failed = []

    metadata = {
        "center_id": center_id,
        "available_sequences": [],
        "missing_sequences": [],
        "usable": False,
        "sequences": {},
        "convert_time": str(datetime.now()),
        "source_dir": str(raw_dir)
    }

    # =====================================================
    # 转换
    # =====================================================

    for seq_dir in seq_dirs:

        seq_type = identify_sequence(seq_dir.name)

        # =================================================
        # 未识别
        # =================================================

        if seq_type is None:

            logger.warning(
                f"[UNRECOGNIZED] {center_id} : {seq_dir.name}"
            )

            continue

        logger.info(f"{seq_dir.name} -> {seq_type}")

        output_path = out_dir / f"{seq_type}.nii.gz"

        # =================================================
        # 防止覆盖
        # =================================================

        if output_path.exists():

            logger.warning(
                f"[SKIP EXISTS] "
                f"{output_path.name} already exists, "
                f"skip: {seq_dir.name}"
            )

            continue

        # =================================================
        # 转换
        # =================================================

        ok = convert_dicom_to_nifti(
            seq_dir,
            output_path,
            seq_type
        )

        if ok:

            converted.append(seq_type)

            metadata["available_sequences"].append(seq_type)

            info = get_nii_info(output_path)

            metadata["sequences"][seq_type] = {
                "file": output_path.name,
                "shape": info["shape"],
                "spacing": info["spacing"],
                "origin_folder": seq_dir.name
            }

        else:

            failed.append(seq_type)

    # =====================================================
    # 去重
    # =====================================================

    metadata["available_sequences"] = list(
        sorted(set(metadata["available_sequences"]))
    )

    # =====================================================
    # 缺失序列
    # =====================================================

    metadata["missing_sequences"] = [
        s for s in ALL_SEQUENCES
        if s not in metadata["available_sequences"]
    ]

    # =====================================================
    # companion 检查
    # =====================================================

    has_companion = any([
        "DixonF" in metadata["available_sequences"],
        "DixonIP" in metadata["available_sequences"],
        "DixonOP" in metadata["available_sequences"]
    ])

    # =====================================================
    # usable
    # =====================================================

    metadata["usable"] = all([
        "T2FS" in metadata["available_sequences"],
        "DWI" in metadata["available_sequences"],
        "ADC" in metadata["available_sequences"],
        "DixonW" in metadata["available_sequences"],
        has_companion,
        "DynT1W" in metadata["available_sequences"]
    ])

    # =====================================================
    # 保存 metadata
    # =====================================================

    meta_path = out_dir / "metadata.json"

    with open(meta_path, "w", encoding="utf-8") as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    return {
        "center_id": center_id,
        "converted": converted,
        "failed": failed
    }

# =========================================================
# batch
# =========================================================

def batch_convert(raw_root, nii_root, cases=None):

    if cases is None:

        centers = [
            d.name for d in raw_root.iterdir()
            if d.is_dir()
        ]

    else:

        centers = cases

    logger.info(f"病例数: {len(centers)}")

    with open(FAILED_CASES_FILE, "w") as f:
        f.write("failed cases\n")

    for i, c in enumerate(centers):

        logger.info("=" * 60)
        logger.info(f"[{i+1}/{len(centers)}] {c}")

        res = process_one_center(
            c,
            raw_root,
            nii_root
        )

        if res["failed"]:

            with open(FAILED_CASES_FILE, "a") as f:
                f.write(f"{c}: {res['failed']}\n")

# =========================================================
# main
# =========================================================

if __name__ == "__main__":

    RAW = Path(os.environ.get("RAW_DICOM_ROOT", "./data/raw_dicom"))
    OUT = Path(os.environ.get("NIFTI_ROOT", "./data/nii"))

    # 单病例测试
    # batch_convert(
    #     RAW,
    #     OUT,
    #     cases=["Center2-13469885"]
    # )

    # 全量运行
    batch_convert(RAW, OUT)
