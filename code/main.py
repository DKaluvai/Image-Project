import os
import sys
import cv2
import numpy as np
import torch
import open3d as o3d

# ─────────────────────────────────────────────
# Add local project paths
LAMA_PATH = os.path.join(os.getcwd(), "lama")
PIFU_PATH = os.path.join(os.getcwd(), "pifuhd")
sys.path.extend([LAMA_PATH, PIFU_PATH])

# ─────────────────────────────────────────────
# Utility: Ensure directories exist
def ensure_dirs():
    for folder in ["input_images", "input_3d", "output"]:
        os.makedirs(folder, exist_ok=True)

# ─────────────────────────────────────────────
# COLOR UNIFORMITY FUNCTION
def color_uniformity(src_img, ref_img):
    """Match colors of src_img to ref_img using LAB histogram matching."""
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    matched = src_lab.copy()

    for i in range(3):
        src_hist, _ = np.histogram(src_lab[:,:,i].ravel(), 256, [0,256])
        ref_hist, _ = np.histogram(ref_lab[:,:,i].ravel(), 256, [0,256])
        cdf_src = np.cumsum(src_hist).astype(float) / src_hist.sum()
        cdf_ref = np.cumsum(ref_hist).astype(float) / ref_hist.sum()
        lookup = np.interp(cdf_src, cdf_ref, np.arange(256))
        matched[:,:,i] = cv2.LUT(src_lab[:,:,i], lookup.astype(np.uint8))

    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────
# 2D → 2D completion (LaMa)
def complete_2d(img_path):
    print("🧠 Running 2D image completion using LaMa...")
    try:
        from lama.saicinpainting.training.trainers import load_checkpoint
        from lama.saicinpainting.evaluation.utils import move_to_device, pad_img_to_modulo
    except ModuleNotFoundError:
        print("❌ LaMa module not found. Make sure you have cloned it properly:")
        print("   git clone https://github.com/advimman/lama.git")
        sys.exit(1)

    ckpt_path = os.path.join(LAMA_PATH, "checkpoints", "big-lama.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("Download from: https://github.com/advimman/lama#model-zoo")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(ckpt_path, strict=False)
    model = move_to_device(model, device)
    model.eval()

    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = move_to_device(tensor, device)

    mask = torch.zeros_like(tensor)
    h, w = mask.shape[2], mask.shape[3]
    mask[:,:,h//3:h//3*2, w//3:w//3*2] = 1.0

    tensor = pad_img_to_modulo(tensor, 8)
    mask = pad_img_to_modulo(mask, 8)

    with torch.no_grad():
        output = model(tensor, mask)
    output = (output.clamp(0,1).cpu()[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    uniform = color_uniformity(result, img)
    return uniform

# ─────────────────────────────────────────────
# 2D → 3D reconstruction (PIFuHD)
def reconstruct_3d(img_path):
    print("🔄 Running 3D reconstruction with PIFuHD (this may take a while)...")

    ckpt_path = os.path.join(PIFU_PATH, "checkpoints", "pifuhd.pt")
    if not os.path.exists(ckpt_path):
        print(f"❌ Missing PIFuHD checkpoint: {ckpt_path}")
        print("Download from: https://github.com/facebookresearch/pifuhd")
        sys.exit(1)

    os.system(
        f'python "{os.path.join(PIFU_PATH, "apps/simple_test.py")}" '
        f'-r 512 --use_rect -i "{img_path}" -o "output" --ckpt_path "{ckpt_path}"'
    )

    out_dir = os.path.join("output", "pifuhd_final")
    if not os.path.exists(out_dir):
        print("❌ PIFuHD output not found. Check PIFuHD logs.")
        return None

    for file in os.listdir(out_dir):
        if file.endswith(".obj") or file.endswith(".ply"):
            path = os.path.join(out_dir, file)
            print(f"✅ 3D model generated: {path}")
            return path

    print("❌ No 3D model file found.")
    return None

# ─────────────────────────────────────────────
# 3D → 3D color uniformity
def recolor_3d(src_ply, ref_img):
    print("🎨 Applying color uniformity to 3D model...")
    pc = o3d.io.read_point_cloud(src_ply)
    colors = np.asarray(pc.colors)
    if colors.size == 0:
        print("⚠️ No colors found in the point cloud. Skipping recoloring.")
        return pc
    ref = cv2.imread(ref_img)
    ref = cv2.resize(ref, (colors.shape[0], 1)).reshape(-1, 3) / 255.0
    pc.colors = o3d.utility.Vector3dVector(ref)
    return pc

# ─────────────────────────────────────────────
# MAIN FUNCTION
def main():
    ensure_dirs()
    print("Select task:")
    print("1 - 2D→2D completion")
    print("2 - 2D→3D reconstruction")
    print("3 - 3D→3D color uniformity")
    choice = input("Enter 1 / 2 / 3: ")

    if choice == "1":
        img = os.path.join("input_images", os.listdir("input_images")[0])
        result = complete_2d(img)
        out = os.path.join("output", "2d_completed.png")
        cv2.imwrite(out, result)
        print(f"✅ 2D completed image saved at: {out}")

    elif choice == "2":
        img = os.path.join("input_images", os.listdir("input_images")[0])
        temp = os.path.join("output", "temp_completed.png")
        cv2.imwrite(temp, complete_2d(img))
        reconstruct_3d(temp)

    elif choice == "3":
        ply = os.path.join("input_3d", os.listdir("input_3d")[0])
        ref = os.path.join("input_images", os.listdir("input_images")[0])
        pc = recolor_3d(ply, ref)
        out = os.path.join("output", "3d_color_uniform.ply")
        o3d.io.write_point_cloud(out, pc)
        print(f"✅ 3D color-uniform model saved at: {out}")

    else:
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
