# 512
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import traceback
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# =========================================================
# paths
# =========================================================

INPUT_DIR = os.environ.get(
    "NORMALIZED_ROOT_512",
    "./data/cropped_normalized_512",
)
OUTPUT_DIR = os.environ.get(
    "LR_HR_ROOT_512",
    "./data/lr_hr_512",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# random seed
# =========================================================

SEED = 2026

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# logs
# =========================================================

CSV_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "lr_hr_generation_log.csv"
)

ERROR_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "lr_hr_error_log.txt"
)


# =========================================================
# load existing log
# resume support
# =========================================================

if os.path.exists(CSV_LOG_PATH):

    old_df = pd.read_csv(CSV_LOG_PATH)

    completed = set(
        zip(
            old_df["case_id"].astype(str),
            old_df["scale"].astype(int)
        )
    )

    all_logs = old_df.to_dict("records")

    print(
        f"[INFO] loaded existing log: "
        f"{len(completed)} finished items"
    )

else:

    completed = set()
    all_logs = []


# =========================================================
# gaussian blur
# =========================================================

def gaussian_blur(img, sigma):

    return sitk.DiscreteGaussian(
        img,
        variance=sigma ** 2
    )


# =========================================================
# downsample
# =========================================================

def downsample_image(img, scale):

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_spacing = (
        original_spacing[0] * scale,
        original_spacing[1] * scale,
        original_spacing[2]
    )

    new_size = (
        max(1, int(original_size[0] / scale)),
        max(1, int(original_size[1] / scale)),
        original_size[2]
    )

    resampler = sitk.ResampleImageFilter()

    # =====================================================
    # Linear 比 BSpline 快很多
    # =====================================================

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampler.SetOutputSpacing(
        new_spacing
    )

    resampler.SetSize(
        new_size
    )

    resampler.SetOutputDirection(
        img.GetDirection()
    )

    resampler.SetOutputOrigin(
        img.GetOrigin()
    )

    resampler.SetOutputPixelType(
        sitk.sitkFloat32
    )

    return resampler.Execute(img)


# =========================================================
# upsample to HR grid
# =========================================================

def upsample_to_hr(lr_img, hr_img):

    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(hr_img)

    # =====================================================
    # Linear 更快
    # =====================================================

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampler.SetOutputPixelType(
        sitk.sitkFloat32
    )

    return resampler.Execute(lr_img)


# =========================================================
# add noise
# =========================================================

def add_noise(img, noise_std=0.01):

    arr = sitk.GetArrayFromImage(img).astype(
        np.float32
    )

    noise = np.random.normal(
        0,
        noise_std,
        arr.shape
    ).astype(np.float32)

    arr = arr + noise

    noisy_img = sitk.GetImageFromArray(arr)

    noisy_img.CopyInformation(img)

    return noisy_img


# =========================================================
# visualization
# =========================================================

def save_visualization(hr, lr_dict, save_path):

    hr_arr = sitk.GetArrayFromImage(hr)

    z = hr_arr.shape[0] // 2

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(
        hr_arr[z],
        cmap='gray'
    )

    axes[0].set_title("HR")

    scales = [2, 3, 4]

    for i, scale in enumerate(scales):

        if scale not in lr_dict:
            continue

        lr_arr = sitk.GetArrayFromImage(
            lr_dict[scale]
        )

        axes[i + 1].imshow(
            lr_arr[z],
            cmap='gray'
        )

        axes[i + 1].set_title(
            f"LR x{scale}"
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches='tight'
    )

    plt.close()


# =========================================================
# save csv
# =========================================================

def save_csv_log():

    df = pd.DataFrame(all_logs)

    df.to_csv(
        CSV_LOG_PATH,
        index=False
    )


# =========================================================
# error log
# =========================================================

def write_error_log(msg):

    with open(ERROR_LOG_PATH, "a") as f:

        f.write(msg)
        f.write("\n")


# =========================================================
# check complete case
# =========================================================

def is_case_completed(case_out_dir):

    hr_path = os.path.join(
        case_out_dir,
        "HR_T2FS.nii.gz"
    )

    lr2 = os.path.join(
        case_out_dir,
        "LR_T2FS_x2.nii.gz"
    )

    lr3 = os.path.join(
        case_out_dir,
        "LR_T2FS_x3.nii.gz"
    )

    lr4 = os.path.join(
        case_out_dir,
        "LR_T2FS_x4.nii.gz"
    )

    return (
        os.path.exists(hr_path)
        and os.path.exists(lr2)
        and os.path.exists(lr3)
        and os.path.exists(lr4)
    )


# =========================================================
# main
# =========================================================

cases = sorted(os.listdir(INPUT_DIR))

for idx, case_id in enumerate(cases):

    print("\n================================================")
    print(f"[{idx+1}/{len(cases)}] {case_id}")
    print("================================================")

    case_in_dir = os.path.join(
        INPUT_DIR,
        case_id
    )

    case_out_dir = os.path.join(
        OUTPUT_DIR,
        case_id
    )

    # =====================================================
    # 整个case已经完成
    # 直接跳过
    # =====================================================

    if is_case_completed(case_out_dir):

        print("[SKIP] already completed")

        continue

    # =====================================================
    # input HR
    # =====================================================

    t2_path = os.path.join(
        case_in_dir,
        "T2FS_crop.nii.gz"
    )

    if not os.path.exists(t2_path):

        print("[WARN] missing T2FS")

        write_error_log(
            f"[MISSING] {case_id} no T2FS"
        )

        continue

    try:

        os.makedirs(
            case_out_dir,
            exist_ok=True
        )

        # =================================================
        # read HR
        # =================================================

        hr_img = sitk.ReadImage(t2_path)

        # =================================================
        # save HR
        # =================================================

        hr_save_path = os.path.join(
            case_out_dir,
            "HR_T2FS.nii.gz"
        )

        if not os.path.exists(hr_save_path):

            sitk.WriteImage(
                hr_img,
                hr_save_path
            )

        # =================================================
        # LR dict
        # =================================================

        lr_dict = {}

        # =================================================
        # scales
        # =================================================

        for scale in [2, 3, 4]:

            lr_save_path = os.path.join(
                case_out_dir,
                f"LR_T2FS_x{scale}.nii.gz"
            )

            # -------------------------------------------------
            # 文件已存在
            # 直接跳过
            # -------------------------------------------------

            if os.path.exists(lr_save_path):

                print(f"[SKIP] x{scale} exists")

                try:

                    lr_dict[scale] = sitk.ReadImage(
                        lr_save_path
                    )

                except:
                    pass

                continue

            # -------------------------------------------------
            # log已存在
            # 直接跳过
            # -------------------------------------------------

            if (case_id, scale) in completed:

                print(f"[SKIP] x{scale} logged")

                continue

            print(f"[INFO] generating x{scale}")

            try:

                # =============================================
                # blur
                # =============================================

                sigma = random.uniform(
                    0.8,
                    1.5
                )

                blur_img = gaussian_blur(
                    hr_img,
                    sigma
                )

                # =============================================
                # downsample
                # =============================================

                low_img = downsample_image(
                    blur_img,
                    scale
                )

                # =============================================
                # upsample
                # =============================================

                lr_img = upsample_to_hr(
                    low_img,
                    hr_img
                )

                # =============================================
                # noise
                # =============================================

                noise_std = random.uniform(
                    0.003,
                    0.015
                )

                lr_img = add_noise(
                    lr_img,
                    noise_std
                )

                # =============================================
                # save
                # =============================================

                sitk.WriteImage(
                    lr_img,
                    lr_save_path
                )

                lr_dict[scale] = lr_img

                # =============================================
                # log
                # =============================================

                log_item = {

                    "case_id": case_id,

                    "scale": scale,

                    "blur_sigma": sigma,

                    "noise_std": noise_std,

                    "downsample_method": "Linear",

                    "random_seed": SEED
                }

                all_logs.append(log_item)

                completed.add(
                    (case_id, scale)
                )

                print(f"[INFO] x{scale} done")

            except Exception as scale_error:

                err_msg = (
                    f"\n[ERROR] case={case_id} "
                    f"x{scale}\n"
                    f"{str(scale_error)}\n"
                    f"{traceback.format_exc()}\n"
                )

                print(err_msg)

                write_error_log(err_msg)

        # =================================================
        # 每个case结束后再保存csv
        # 大幅减少IO
        # =================================================

        save_csv_log()

        # =================================================
        # visualization
        # 只抽查5%
        # =================================================

        vis_path = os.path.join(
            case_out_dir,
            "LR_HR_QC.png"
        )

        if (
            len(lr_dict) > 0
            and random.random() < 0.05
        ):

            try:

                save_visualization(
                    hr_img,
                    lr_dict,
                    vis_path
                )

            except Exception as vis_error:

                err_msg = (
                    f"\n[VIS ERROR] {case_id}\n"
                    f"{str(vis_error)}\n"
                    f"{traceback.format_exc()}\n"
                )

                print(err_msg)

                write_error_log(err_msg)

        print("[INFO] case done")

    except Exception as e:

        err_msg = (
            f"\n[FATAL ERROR] {case_id}\n"
            f"{str(e)}\n"
            f"{traceback.format_exc()}\n"
        )

        print(err_msg)

        write_error_log(err_msg)


# =========================================================
# final save
# =========================================================

save_csv_log()

print("\n================================================")
print("[DONE] LR-HR generation finished")
print("================================================")
