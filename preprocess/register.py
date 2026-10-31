#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import SimpleITK as sitk


# =========================================================
# CPU线程（重要）
# =========================================================
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(16)


# =========================================================
# 统一方向
# =========================================================
def orient_to_ras(img):
    return sitk.DICOMOrient(img, "RAS")


# =========================================================
# 快速下采样（配准专用）
# 巨大提速
# =========================================================
def shrink_for_registration(img, factor=4):

    return sitk.Shrink(
        img,
        [factor, factor, 1]
    )


# =========================================================
# 配准
# =========================================================
def register_to_t2fs(fixed_path, moving_path, output_path, seq_name=None):

    print(f"\n[INFO] Registering:")
    print(f"  Moving : {moving_path}")

    # =========================
    # 读取
    # =========================
    fixed = sitk.ReadImage(
        fixed_path,
        sitk.sitkFloat32
    )

    moving = sitk.ReadImage(
        moving_path
    )

    fixed = orient_to_ras(fixed)

    # =========================
    # 4D处理
    # =========================
    is_4d = False

    if moving.GetDimension() == 4:

        is_4d = True

        print(f"  [WARN] 4D detected in {seq_name}")

        size = list(moving.GetSize())

        index = [0, 0, 0, size[3] - 1]

        size[3] = 0

        moving = sitk.Extract(
            moving,
            size,
            index
        )

    moving = sitk.Cast(
        moving,
        sitk.sitkFloat32
    )

    moving = orient_to_ras(moving)

    # =====================================================
    # 低分辨率配准（核心提速）
    # =====================================================
    fixed_low = shrink_for_registration(
        fixed,
        factor=4
    )

    moving_low = shrink_for_registration(
        moving,
        factor=4
    )

    # =====================================================
    # 初始化
    # =====================================================
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_low,
        moving_low,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    affine_transform = sitk.AffineTransform(3)

    affine_transform.SetCenter(
        initial_transform.GetCenter()
    )

    # =====================================================
    # registration
    # =====================================================
    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(32)

    registration_method.SetMetricSamplingStrategy(
        registration_method.RANDOM
    )

    # 提速
    registration_method.SetMetricSamplingPercentage(0.05)

    registration_method.SetInterpolator(
        sitk.sitkLinear
    )

    # 提速
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=50,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 多分辨率
    registration_method.SetShrinkFactorsPerLevel(
        [4, 2, 1]
    )

    registration_method.SetSmoothingSigmasPerLevel(
        [2, 1, 0]
    )

    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(
        affine_transform,
        inPlace=False
    )

    # =====================================================
    # 执行（低分辨率）
    # =====================================================
    final_transform = registration_method.Execute(
        fixed_low,
        moving_low
    )

    metric = registration_method.GetMetricValue()

    print(f"  [INFO] Metric: {metric}")

    # =====================================================
    # 原图resample
    # =====================================================
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    sitk.WriteImage(
        registered,
        output_path
    )

    print(f"  [INFO] Saved")

    return {

        "metric": metric,

        "fixed_size": fixed.GetSize(),
        "moving_size": moving.GetSize(),
        "registered_size": registered.GetSize(),

        "fixed_spacing": fixed.GetSpacing(),
        "moving_spacing": moving.GetSpacing(),
        "registered_spacing": registered.GetSpacing(),

        "is_4d": is_4d
    }


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":

    input_root = os.environ.get("NIFTI_ROOT", "./data/nii")

    output_root = os.environ.get("REGISTERED_ROOT", "./data/registered_to_T2FS")

    os.makedirs(output_root, exist_ok=True)

    moving_sequences = [
        "DixonW",
        "DixonF",
        "DixonIP",
        "DixonOP",
        "DWI",
        "ADC",
        "DynT1W"
    ]

    success = 0
    failed = 0
    skipped = 0

    report_rows = []

    cases = sorted(os.listdir(input_root))

    cases = [
        c for c in cases
        if os.path.isdir(os.path.join(input_root, c))
    ]

    total_cases = len(cases)

    # =====================================================
    # 遍历病例
    # =====================================================
    for i, case_id in enumerate(cases):

        print("\n================================================")
        print(f"[{i+1}/{total_cases}] {case_id}")
        print("================================================")

        case_input_dir = os.path.join(
            input_root,
            case_id
        )

        fixed_path = os.path.join(
            case_input_dir,
            "T2FS.nii.gz"
        )

        if not os.path.exists(fixed_path):

            print("[SKIP] Missing T2FS")

            continue

        case_output_dir = os.path.join(
            output_root,
            case_id
        )

        os.makedirs(case_output_dir, exist_ok=True)

        # =================================================
        # 序列循环
        # =================================================
        for seq in moving_sequences:

            moving_path = os.path.join(
                case_input_dir,
                f"{seq}.nii.gz"
            )

            if not os.path.exists(moving_path):
                continue

            output_path = os.path.join(
                case_output_dir,
                f"{seq}_to_T2FS.nii.gz"
            )

            if os.path.exists(output_path):

                skipped += 1

                continue

            try:

                info = register_to_t2fs(
                    fixed_path,
                    moving_path,
                    output_path,
                    seq_name=seq
                )

                success += 1

                report_rows.append({

                    "case_id": case_id,
                    "sequence": seq,

                    "success": True,

                    "metric": info["metric"],

                    "fixed_size": str(info["fixed_size"]),
                    "moving_size": str(info["moving_size"]),
                    "registered_size": str(info["registered_size"]),

                    "fixed_spacing": str(info["fixed_spacing"]),
                    "moving_spacing": str(info["moving_spacing"]),
                    "registered_spacing": str(info["registered_spacing"]),

                    "is_4d": info["is_4d"],

                    "error_message": ""
                })

            except Exception as e:

                failed += 1

                print(f"[ERROR] {case_id} - {seq}")
                print(e)

                report_rows.append({

                    "case_id": case_id,
                    "sequence": seq,

                    "success": False,

                    "metric": None,

                    "fixed_size": "",
                    "moving_size": "",
                    "registered_size": "",

                    "fixed_spacing": "",
                    "moving_spacing": "",
                    "registered_spacing": "",

                    "is_4d": "",

                    "error_message": str(e)
                })

    # =====================================================
    # 保存报告
    # =====================================================
    df = pd.DataFrame(report_rows)

    xlsx_path = os.path.join(
        output_root,
        "registration_report.xlsx"
    )

    df.to_excel(
        xlsx_path,
        index=False
    )

    print("\n================================================")
    print("ALL DONE")
    print("================================================")

    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")

    print("\nReport:")
    print(xlsx_path)
