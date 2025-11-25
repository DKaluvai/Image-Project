import os
import cv2
import numpy as np
import torch
import open3d as o3d
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from simple_lama_inpainting import inpaint

# -----------------------------
# Utility functions
# -----------------------------
def lab_histogram_match(source_img, reference_img):
    """Color uniformity using Lab histogram matching"""
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    matched = source_lab.copy()
    for i in range(3):
        src_hist, bins = np.histogram(source_lab[:,:,i].flatten(), 256, [0,256])
        ref_hist, _ = np.histogram(reference_lab[:,:,i].flatten(), 256, [0,256])
        cdf_src = np.cumsum(src_hist).astype(float)/source_lab[:,:,i].size
        cdf_ref = np.cumsum(ref_hist).astype(float)/reference_lab[:,:,i].size
        lookup = np.interp(cdf_src, cdf_ref, np.arange(256))
        matched[:,:,i] = cv2.LUT(source_lab[:,:,i], lookup.astype(np.uint8))
    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

def split_image(img, rows=3, cols=3):
    """Split image into tiles"""
    h, w = img.shape[:2]
    tiles = []
    tile_h, tile_w = h//rows, w//cols
    for i in range(rows):
        for j in range(cols):
            tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    return tiles, tile_h, tile_w

def recombine_tiles(tiles, rows=3, cols=3):
    """Recombine tiles into image"""
    row_imgs = []
    for i in range(rows):
        row_imgs.append(np.hstack(tiles[i*cols:(i+1)*cols]))
    return np.vstack(row_imgs)

def auto_reorder_tiles(tiles):
    """Simple automatic reordering using mean color similarity (approximation)"""
    tiles_sorted = sorted(tiles, key=lambda x: np.mean(x))
    return tiles_sorted

def inpaint_seams(img):
    """Simple seam inpainting using edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(gray,50,150)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def display_image(img, title="Image"):
    """Display image using OpenCV"""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_point_cloud(rgb_img, depth_map):
    """Generate colored point cloud from RGB and depth"""
    h, w = depth_map.shape
    fx = fy = 1  # Focal length normalized
    cx, cy = w//2, h//2
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            z = depth_map[v,u]
            if z==0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x,y,z])
            colors.append(rgb_img[v,u]/255.0)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pc

# -----------------------------
# 2D Completion using LaMa
# -----------------------------
def complete_2d_image(img):
    """Complete 2D image using LaMa inpainting"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[100:200, 100:200] = 255  # Example mask
    inpainted_img = inpaint(img, mask)
    uniform_img = lab_histogram_match(inpainted_img, img)
    return uniform_img

# -----------------------------
# 2D → 3D Reconstruction using PIFuHD
# -----------------------------
def reconstruct_3d_from_2d(img):
    """Full 3D reconstruction from single 2D image using PIFuHD"""
    # Placeholder for PIFuHD integration
    return generate_point_cloud(img, np.ones_like(img[:,:,0]))

# -----------------------------
# 3D → 3D Color Uniformity
# -----------------------------
def color_uniformity_3d(source_pc, reference_img):
    """Transfer color from reference 2D to 3D point cloud"""
    colors = np.asarray(source_pc.colors)
    ref_resized = cv2.resize(reference_img, (colors.shape[0],1))
    ref_colors = ref_resized.reshape(-1,3)/255.0
    colors = ref_colors
    source_pc.colors = o3d.utility.Vector3dVector(colors)
    return source_pc

# -----------------------------
# Main script
# -----------------------------
def main():
    print("Select output type:")
    print("1 - 2D → 2D completion")
    print("2 - 2D → 3D reconstruction")
    print("3 - 3D → 3D color uniformity")
    choice = input("Enter choice [1/2/3]: ")

    if choice == "1":
        input_path = os.path.join("input_images", os.listdir("input_images")[0])
        img = cv2.imread(input_path)
        completed = complete_2d_image(img)
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "2d_completed.png")
        cv2.imwrite(out_path, completed)
        print(f"2D completed image saved at {out_path}")
        display_image(completed, "2D Completion")

    elif choice == "2":
        input_path = os.path.join("input_images", os.listdir("input_images")[0])
        img = cv2.imread(input_path)
        completed = complete_2d_image(img)
        pc = reconstruct_3d_from_2d(completed)
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "3d_reconstructed.ply")
        o3d.io.write_point_cloud(out_path, pc)
        print(f"3D reconstructed point cloud saved at {out_path}")
        o3d.visualization.draw_geometries([pc])

    elif choice == "3":
        source_path = os.path.join("input_3d", os.listdir("input_3d")[0])
        reference_path = os.path.join("input_images", os.listdir("input_images")[0])
        source_pc = o3d.io.read_point_cloud(source_path)
        reference_img = cv2.imread(reference_path)
        uniform_pc = color_uniformity_3d(source_pc, reference_img)
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "3d_color_uniform.ply")
        o3d.io.write_point_cloud(out_path, uniform_pc)
        print(f"3D color-uniform point cloud saved at {out_path}")
        o3d.visualization.draw_geometries([uniform_pc])

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
