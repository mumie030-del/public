import os
from os.path import join

import numpy as np
from PIL import Image


DATA_ROOT = "data3"        # 原始数据根目录（你的 data3）
OUT_DIR = "data/npy"       # 输出 npy 目录（会自动创建）
NUM_CHANNELS = 26          # 每个病例应有的帧数
TARGET_SIZE = (256, 256)   # 统一 resize 尺寸，和训练用的一致


def _collect_numeric_images(search_dir: str):
    """
    收集形如 101.jpg / 603.png 这类“纯数字文件名”的图片序列，
    并按数字从小到大排序返回完整路径列表。
    """
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    candidates = []
    for fn in os.listdir(search_dir):
        fp = join(search_dir, fn)
        if not os.path.isfile(fp):
            continue
        lower = fn.lower()
        if not lower.endswith(exts):
            continue
        stem = os.path.splitext(fn)[0]
        if stem.isdigit():
            candidates.append((int(stem), fp))
    candidates.sort(key=lambda x: x[0])
    return [fp for _, fp in candidates]


def case_dir_iter(data_root: str):
    """遍历 data_root 下的病例文件夹（纯数字命名的目录，按编号升序）"""
    dirs = []
    for d in os.listdir(data_root):
        full = join(data_root, d)
        if os.path.isdir(full) and d.isdigit():
            dirs.append((int(d), full))
    dirs.sort(key=lambda x: x[0])
    for idx, path in dirs:
        yield idx, path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    num_ok, num_skip = 0, 0

    for case_idx, folder_path in case_dir_iter(DATA_ROOT):
        # 兼容两种结构：
        # 1) folder/images/*.jpg
        # 2) folder/*.jpg (扁平目录)
        images_dir = join(folder_path, "images")
        search_dir = images_dir if os.path.isdir(images_dir) else folder_path

        image_files = _collect_numeric_images(search_dir)
        if len(image_files) != NUM_CHANNELS:
            print(
                f"[跳过] 病例 {case_idx} 在 {search_dir} 中找到 {len(image_files)} 张有效图片 "
                f"(期望 {NUM_CHANNELS})"
            )
            num_skip += 1
            continue

        img_stack = []
        for p in image_files:
            with Image.open(p) as img_obj:
                # resize 到统一尺寸
                img_obj = img_obj.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
                img_np = np.array(img_obj)
                # 转灰度（与 Data3Dataset 中的逻辑保持一致）
                if img_np.ndim == 3:
                    img_np = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
                img_stack.append(img_np.astype(np.float32))

        # 堆成 (26, H, W)
        images_np = np.stack(img_stack, axis=0)  # (C, H, W)

        # 简单归一化到 0-1（和训练前的数据预处理类似）
        max_val = images_np.max()
        if max_val > 0:
            # 简单假设 8bit / 16bit，按最大值自适应缩放到 0-1
            images_np = images_np / max_val

        # 输出文件名：P001_26frames.npy 这种形式
        patient_id = f"P{case_idx:03d}"
        out_path = join(OUT_DIR, f"{patient_id}_26frames.npy")
        np.save(out_path, images_np)
        num_ok += 1

        print(f"[OK] 病例 {case_idx:03d} -> {out_path}  形状: {images_np.shape}")

    print("\n====== 完成导出 ======")
    print(f"成功导出: {num_ok} 例")
    print(f"被跳过  : {num_skip} 例（帧数不是 {NUM_CHANNELS}）")


if __name__ == "__main__":
    main()


