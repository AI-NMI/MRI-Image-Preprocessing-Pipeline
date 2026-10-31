#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)

# =========================================================
# convert numpy type -> python native type
# =========================================================
def to_python_type(obj):

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


# =========================================================
# liver ROI crop
# =========================================================
def get_liver_crop_box(img, margin=(8, 24, 24)):

    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    sitk_img = sitk.GetImageFromArray(arr)

    otsu_filter = sitk.OtsuThresholdImageFilter()
    mask = otsu_filter.Execute(sitk_img)

    mask_arr = sitk.GetArrayFromImage(mask) > 0

    mask_img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))

    cc = sitk.ConnectedComponent(mask_img)
    relabel = sitk.RelabelComponent(cc)

    largest = sitk.GetArrayFromImage(relabel) == 1

    coords = np.where(largest)

    if len(coords[0]) == 0:
        raise ValueError("Empty body/liver mask")

    z1, z2 = coords[0].min(), coords[0].max()
    y1, y2 = coords[1].min(), coords[1].max()
    x1, x2 = coords[2].min(), coords[2].max()

    mz, my, mx = margin

    z1 = max(z1 - mz, 0)
    y1 = max(y1 - my, 0)
    x1 = max(x1 - mx, 0)

    z2 = min(z2 + mz, arr.shape[0])
    y2 = min(y2 + my, arr.shape[1])
    x2 = min(x2 + mx, arr.shape[2])

    # =====================================================
    # numpy.int64 -> python int
    # =====================================================
    return (
        int(z1),
        int(z2),
        int(y1),
        int(y2),
        int(x1),
        int(x2)
    )


# =========================================================
# crop
# =========================================================
def crop_with_box(img, crop_box):

    arr = sitk.GetArrayFromImage(img)

    z1, z2, y1, y2, x1, x2 = crop_box

    cropped = arr[z1:z2, y1:y2, x1:x2]

    out = sitk.GetImageFromArray(cropped.astype(np.float32))

    spacing = img.GetSpacing()
    direction = img.GetDirection()
    origin = np.array(img.GetOrigin())

    new_origin = origin + np.array([
        x1 * spacing[0],
        y1 * spacing[1],
        z1 * spacing[2]
    ])

    out.SetSpacing(spacing)
    out.SetDirection(direction)
    out.SetOrigin(tuple(new_origin))

    return out


# =========================================================
# foreground mask
# =========================================================
def create_foreground_mask(arr):

    positive = arr[arr > 0]

    if len(positive) == 0:
        raise ValueError("Empty image")

    thresh = np.percentile(positive, 5)

    mask = arr > thresh

    return mask


# =========================================================
# normalize
# =========================================================
def normalize(img, seq_name="MRI"):

    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    mask = create_foreground_mask(arr)

    values = arr[mask]

    if seq_name == "ADC":

        p_low = np.percentile(values, 0.5)
        p_high = np.percentile(values, 99.0)

    else:

        p_low = np.percentile(values, 0.5)
        p_high = np.percentile(values, 99.5)

    arr = np.clip(arr, p_low, p_high)

    mean = arr[mask].mean()
    std = arr[mask].std()

    norm = (arr - mean) / (std + 1e-8)

    bg_value = float(norm[mask].min())

    norm[~mask] = bg_value

    out = sitk.GetImageFromArray(
        norm.astype(np.float32)
    )

    out.CopyInformation(img)

    return out, {
        "percentile_low": float(p_low),
        "percentile_high": float(p_high),
        "mean": float(mean),
        "std": float(std)
    }

# =========================================================
# resize XY only
# =========================================================
def resize_xy(img, target_size=(256,256)):

    size = img.GetSize()
    spacing = img.GetSpacing()

    new_size = [
        target_size[0],
        target_size[1],
        size[2]
    ]

    new_spacing = [
        spacing[0] * size[0] / target_size[0],
        spacing[1] * size[1] / target_size[1],
        spacing[2]
    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(new_size)

    resampler.SetOutputSpacing(
        new_spacing
    )

    resampler.SetOutputOrigin(
        img.GetOrigin()
    )

    resampler.SetOutputDirection(
        img.GetDirection()
    )

    resampler.SetInterpolator(
        sitk.sitkBSpline
    )

    return resampler.Execute(img)

# =========================================================
# QC
# =========================================================
def quality_check(img, norm_img):

    qc = {}

    arr = sitk.GetArrayFromImage(img)
    norm = sitk.GetArrayFromImage(norm_img)

    mask = create_foreground_mask(arr)

    # =====================================================
    # crop ratio
    # =====================================================
    body_ratio = float(mask.sum() / mask.size)

    qc["body_ratio"] = body_ratio

    qc["crop_too_small"] = bool(body_ratio < 0.05)

    # =====================================================
    # normalized stats
    # =====================================================
    norm_values = norm[mask]

    norm_mean = float(norm_values.mean())
    norm_std = float(norm_values.std())

    qc["norm_mean"] = norm_mean
    qc["norm_std"] = norm_std

    qc["nan_exists"] = bool(np.isnan(norm).sum() > 0)

    qc["all_zero"] = bool(np.abs(norm).sum() < 1e-6)

    qc["std_too_small"] = bool(norm_std < 0.3)

    # =====================================================
    # slice check
    # =====================================================
    qc["few_slices"] = bool(arr.shape[0] < 8)

    # =====================================================
    # overall flag
    # =====================================================
    qc["bad_case"] = bool(any([
        qc["crop_too_small"],
        qc["nan_exists"],
        qc["all_zero"],
        qc["std_too_small"],
        qc["few_slices"]
    ]))

    return qc


# =========================================================
# process case
# =========================================================
def process_case(case_id):

    nii_final_root = os.environ.get("NIFTI_ROOT", "./data/nii")
    registered_root = os.environ.get("REGISTERED_ROOT", "./data/registered_to_T2FS")

    normalized_root = os.environ.get(
        "NORMALIZED_ROOT_256",
        "./data/cropped_normalized_256",
    )
    out_dir = os.path.join(normalized_root, case_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # =====================================================
    # skip processed case
    # =====================================================
    qc_path = os.path.join(out_dir, "qc.json")

    if os.path.exists(qc_path):
        print(f"[SKIP] {case_id} already processed")
        return []

    # =====================================================
    #  repair +Center 
    # =====================================================
    center_id = case_id.split("-")[0]

    seq_map = {

        "T2FS": os.path.join(
            nii_final_root,
            case_id,
            "T2FS.nii.gz"
        ),

        "DWI": os.path.join(
            registered_root,
            center_id,
            case_id,
            "DWI_to_T2FS.nii.gz"
        ),

        "DixonOP": os.path.join(
            registered_root,
            center_id,
            case_id,
            "DixonOP_to_T2FS.nii.gz"
        ),

        "DixonIP": os.path.join(
            registered_root,
            center_id,
            case_id,
            "DixonIP_to_T2FS.nii.gz"
        ),

        "DixonF": os.path.join(
            registered_root,
            center_id,
            case_id,
            "DixonF_to_T2FS.nii.gz"
        ),

        "DixonW": os.path.join(
            registered_root,
            center_id,
            case_id,
            "DixonW_to_T2FS.nii.gz"
        ),

        "DynT1W": os.path.join(
            registered_root,
            center_id,
            case_id,
            "DynT1W_to_T2FS.nii.gz"
        ),
    }

    t2_path = seq_map["T2FS"]

    if not os.path.exists(t2_path):
        raise ValueError(f"Missing T2FS: {t2_path}")

    t2_img = sitk.ReadImage(t2_path)

    # =====================================================
    # generate crop box
    # =====================================================
    import time

    print("[INFO] generating body mask...")

    t0 = time.time()

    crop_box = get_liver_crop_box(t2_img)

    print(
    f"[TIME] crop_box generation = "
    f"{time.time()-t0:.2f}s"
    )

    print(f"[INFO] crop_box = {crop_box}")
    
    qc_json = {
        "case_id": str(case_id),
        "crop_box": [int(x) for x in crop_box],
        "sequences": {}
    }

    rows = []

    # =====================================================
    # process each sequence
    # =====================================================
    for seq, path in seq_map.items():

        if not os.path.exists(path):
            print(f"[WARN] missing {seq}")
            continue

        try:

            print(f"[INFO] processing {case_id} {seq}")

            img = sitk.ReadImage(path, sitk.sitkFloat32)

            original_shape = sitk.GetArrayFromImage(img).shape

            cropped = crop_with_box(
                img,
                crop_box
            )

            cropped_shape = sitk.GetArrayFromImage(
                cropped
                ).shape

            norm, stats = normalize(
                cropped,
                seq
                )

            # ? QC������ resize ֮ǰ
            qc_metrics = quality_check(
                cropped,
                norm
                )

            # �� resize
            norm = resize_xy(
                norm,
                target_size=(256, 256)
                )

            resized_shape = sitk.GetArrayFromImage(
                norm
                ).shape


            

            # =================================================
            # save image
            # =================================================
            out_path = os.path.join(
                out_dir,
                f"{seq}_crop.nii.gz"
            )

            sitk.WriteImage(norm, out_path)

            # =================================================
            # excel row
            # =================================================
            row = {

                "case_id": str(case_id),
                "sequence": str(seq),

                "original_shape": str(tuple(original_shape)),
                "cropped_shape": str(tuple(cropped_shape)),
                
                "resized_shape": str(tuple(resized_shape)),

                "spacing": str(tuple([
                    float(x) for x in img.GetSpacing()
                ])),

                "mean": float(stats["mean"]),
                "std": float(stats["std"]),

                "p_low": float(stats["percentile_low"]),
                "p_high": float(stats["percentile_high"]),

                "body_ratio": float(qc_metrics["body_ratio"]),

                "norm_mean": float(qc_metrics["norm_mean"]),
                "norm_std": float(qc_metrics["norm_std"]),

                "few_slices": bool(qc_metrics["few_slices"]),
                "crop_too_small": bool(qc_metrics["crop_too_small"]),

                "nan_exists": bool(qc_metrics["nan_exists"]),
                "all_zero": bool(qc_metrics["all_zero"]),

                "std_too_small": bool(qc_metrics["std_too_small"]),

                "bad_case": bool(qc_metrics["bad_case"]),

                "status": "success"
            }

            rows.append(row)

            qc_json["sequences"][seq] = row

            print(f"[INFO] saved {out_path}")

        except Exception as e:

            print(f"[ERROR] {case_id} {seq}")
            print(e)

            rows.append({
                "case_id": str(case_id),
                "sequence": str(seq),
                "status": "failed",
                "error": str(e)
            })

    # =====================================================
    # save qc json
    # =====================================================
    qc_path = os.path.join(out_dir, "qc.json")

    with open(qc_path, "w") as f:
        json.dump(
            qc_json,
            f,
            indent=4,
            default=to_python_type
        )

    return rows


# =========================================================
# batch
# =========================================================
if __name__ == "__main__":

    input_root = os.environ.get("NIFTI_ROOT", "./data/nii")

    save_root = os.environ.get(
        "NORMALIZED_ROOT_256",
        "./data/cropped_normalized_256",
    )

    os.makedirs(save_root, exist_ok=True)

    case_ids = sorted(os.listdir(input_root))

    print("=" * 60)
    print(f"[INFO] total cases: {len(case_ids)}")
    print("=" * 60)

    success = 0
    failed = 0

    all_rows = []

    for i, case_id in enumerate(case_ids):

        case_dir = os.path.join(input_root, case_id)

        if not os.path.isdir(case_dir):
            continue

        print("\n" + "=" * 60)
        print(f"[{i+1}/{len(case_ids)}] {case_id}")
        print("=" * 60)

        try:

            rows = process_case(case_id)

            all_rows.extend(rows)

            success += 1

        except Exception as e:

            failed += 1

            print(f"[ERROR] failed case: {case_id}")
            print(e)

        print(f"[COUNT] Success={success} | Failed={failed}")

    # =====================================================
    # save excel summary
    # =====================================================
    df = pd.DataFrame(all_rows)

    excel_path = os.path.join(
        save_root,
        "summary.xlsx"
    )

    df.to_excel(excel_path, index=False)

    print("\n" + "=" * 60)
    print("[INFO] all done.")
    print(f"[INFO] summary saved: {excel_path}")
    print("=" * 60)
