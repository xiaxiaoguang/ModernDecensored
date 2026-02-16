import os
import time
import torch
import gc
import numpy as np
import configparser
import shutil
import argparse
import re
import random
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageEnhance, ImageDraw
from pywinauto import Application, Desktop

from scipy.ndimage import label, binary_dilation, center_of_mass 
from diffusers import DiffusionPipeline
from transformers import Sam2Processor, Sam2Model
from utils.inpainter import DecensorInpainter,DecensorInpainter2
from utils.inpainter2 import DecensorInpainterX
from utils.tools import *
import cv2 

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- CLASS 1: MASK MANAGER (Segmentation, HAI Detection, SAM2 Refinement) ---

class DecensorMaskManager:
    def __init__(self, args):
        self.args = args
        self.hai_dir = os.path.dirname(args.hai_path)
        self.hai_mask_folder = os.path.join(self.hai_dir, "decensor_input")
        self.sam_processor = None
        self.sam_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def preprocess_smart_detection(self, img, debug=False):
        """
        Improved Preprocessing: separating 'Artificial Black' from 'Artistic Dark'.
        Uses Fast Local Variance instead of Laplacian for better texture discrimination.
        """
        img_np = np.array(img)
        
        # 1. Working Channel: 'L' from LAB is best for lightness separation
        if img_np.ndim == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
        else:
            l_channel = img_np.copy()

        # --- IMPROVED LOGIC: Fast Local Variance Map ---
        # Laplacian detects edges (high freq). Variance detects "flatness" (statistical spread).
        # We calculate Variance = E[X^2] - (E[X])^2
        
        # Kernel size: 3x3 is tight, 5x5 is safer for ignoring JPEG artifacts
        k_size = (5, 5) 
        
        # Calculate Mean of Image
        mean_l = cv2.boxFilter(l_channel, cv2.CV_32F, k_size)
        
        # Calculate Mean of Squared Image
        sqr_l = cv2.sqrBoxFilter(l_channel, cv2.CV_32F, k_size)
        
        # Variance = Mean(Square) - Square(Mean)
        # We use abs to avoid negative epsilon errors
        variance = np.abs(sqr_l - mean_l**2)
        std_dev = np.sqrt(variance)

        # --- MASK CREATION ---
        # Thresholds need to be tuned based on your 'dark' style
        # Black bars are usually pure (val < 10) and perfectly flat (std < 2.0)
        
        intensity_limit = self.args.black_level if self.args.black_level > 0 else 30
        flatness_limit = 5.0  # Pixels with std_dev < 5.0 are considered 'flat'
        
        is_dark = l_channel < intensity_limit
        is_flat = std_dev < flatness_limit
        
        # Morphological Cleanup:
        # Bars are solid blocks. Noise is scattered.
        # We erode slightly to remove "speckles" of dark noise, then dilate back.
        raw_mask = np.bitwise_and(is_dark, is_flat.astype(bool)).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel) # Removes small white noise
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel) # Closes small holes in bars

        # --- VISUALIZATION (The part you asked for) ---
        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 8))
            
            # 1. Original
            plt.subplot(2, 3, 1)
            plt.title("Original (L Channel)")
            plt.imshow(l_channel, cmap='gray')
            plt.axis('off')

            # 2. Local Variance (Texture Map) - NORMALIZED for visibility
            plt.subplot(2, 3, 2)
            plt.title(f"Texture Map (Std Dev)\nDarker = Flatter")
            # Log scale helps visualize low-contrast texture
            plt.imshow(std_dev, cmap='magma', vmin=0, vmax=20) 
            plt.axis('off')

            # 3. Intensity Map (Threshold Preview)
            plt.subplot(2, 3, 3)
            plt.title(f"Dark Areas (L < {intensity_limit})")
            plt.imshow(is_dark, cmap='gray')
            plt.axis('off')

            # 4. Flatness Map (Threshold Preview)
            plt.subplot(2, 3, 4)
            plt.title(f"Flat Areas (Std < {flatness_limit})")
            plt.imshow(is_flat, cmap='gray')
            plt.axis('off')

            # 5. Final Intersection (The Mask)
            plt.subplot(2, 3, 5)
            plt.title("Final Intersection Mask")
            plt.imshow(clean_mask, cmap='gray')
            plt.axis('off')

            # 6. Result Preview
            debug_preview = l_channel.copy()
            # Highlight mask in bright green
            debug_preview = cv2.cvtColor(debug_preview, cv2.COLOR_GRAY2RGB)
            debug_preview[clean_mask > 0] = [0, 255, 0]
            
            plt.subplot(2, 3, 6)
            plt.title("Result Overlay")
            plt.imshow(debug_preview)
            plt.axis('off')

            plt.tight_layout()
            plt.show() # Blocks execution until you close the window

        # --- RENDERING (CLAHE + Black Crush) ---
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_channel)
        
        # Crush the masked area to pure black
        l_enhanced[clean_mask > 0] = 0
        
        # Reconstruct
        if img_np.ndim == 3:
            merged = cv2.merge((l_enhanced, a, b))
            output_np = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            output_np = l_enhanced

        return Image.fromarray(output_np)
    
    def load_sam2(self):
        if self.sam_model is None:
            print(f"正在从本地路径加载 SAM2.1 模型: {SAM2_MODEL_PATH}")
            self.sam_processor = Sam2Processor.from_pretrained(SAM2_MODEL_PATH)
            self.sam_model = Sam2Model.from_pretrained(SAM2_MODEL_PATH).to(self.device)
            self.sam_model.eval()
            print(f"SAM2.1 加载完成，运行设备: {self.device}")

    def unload_sam2(self):
        self.sam_model = None
        self.sam_processor = None
        cleanup_gpu()

    def update_hai_config(self, input_dir):
        config_path = os.path.join(self.hai_dir, "hconfig.ini")
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            if 'Paths' not in config: config['Paths'] = {}
            config['Paths']['input'] = input_dir
            with open(config_path, 'w') as f: config.write(f)
            
    def segment_images_to_temp(self):
        print(f"模式: Segmentation - 正在准备带固定偏置缓冲的 HAI 检测块 (Tile Size: {LOGICAL_TILE_W}x{LOGICAL_TILE_H})...")
        if os.path.exists(self.args.temp_tiles): shutil.rmtree(self.args.temp_tiles)
        os.makedirs(self.args.temp_tiles)
        
        # Setup a debug folder to output the masks and distance heatmaps
        debug_dir = os.path.join(self.args.input, "debug_black_distances")
        os.makedirs(debug_dir, exist_ok=True)
        
        files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        for i, f in enumerate(files):
            orig_path = os.path.join(self.args.input, f)
            img = smart_resize(Image.open(orig_path).convert("RGB"))
            img = self.preprocess_smart_detection(img)
            img_np = np.array(img)
            
            # 1. Create a binary mask of pure black pixels ([0, 0, 0])
            is_black = np.all(img_np == [0, 0, 0], axis=-1)
            black_mask = np.uint8(is_black) * 255
            
            # Save the raw black pixel mask
            cv2.imwrite(os.path.join(debug_dir, f"{i}_black_mask.png"), black_mask)
            
            # 2. Distance Transform
            # cv2.distanceTransform calculates the distance to the nearest ZERO pixel.
            # We invert the mask so black pixels = 0, and everything else = 255.
            inverted_mask = cv2.bitwise_not(black_mask)
            
            # Calculate L2 (Euclidean) distance
            dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
            
            # Normalize the distance map to 0-255 for visualization
            dist_visual = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
            dist_visual = np.uint8(dist_visual)
            
            # Apply a JET colormap (Red = far from black, Blue = close to black/is black)
            heatmap = cv2.applyColorMap(dist_visual, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(debug_dir, f"{i}_distance_heatmap.png"), heatmap)
            # ---------------------------------------------------------

            w, h = img.size
            tile_configs = get_tiles_with_padding((w, h))
            for config in tile_configs:
                p_box, l_box = config['padded'], config['logical']
                tile = extract_padded_tile(img, p_box)
                tile_name = f"{i}_T_{l_box[0]}_{l_box[1]}_{l_box[2]}_{l_box[3]}_P_{p_box[0]}_{p_box[1]}_{p_box[2]}_{p_box[3]}.png"
                tile.save(os.path.join(self.args.temp_tiles, tile_name))

    def segment_focused_from_coarse(self):
        print("模式: Segmentation (Focused) - 基于初步检测结果生成重点切片...")
        if os.path.exists(self.args.temp_tiles): shutil.rmtree(self.args.temp_tiles)
        os.makedirs(self.args.temp_tiles)

        files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

        for idx, orig_file in enumerate(files):
            orig_path = os.path.join(self.args.input, orig_file)
            img = smart_resize(Image.open(orig_path).convert("RGB"))
            
            # if self.args.black_level > 0:
            img = self.preprocess_smart_detection(img)
                
            w, h = img.size
            
            full_mask = Image.new("L", (w, h), 0)
            if os.path.exists(self.hai_mask_folder):
                for m_file in os.listdir(self.hai_mask_folder):
                    if m_file.startswith(f"{idx}_T_") and m_file.endswith(".png"):
                        parts = m_file.replace(".png", "").split("_")
                        try:
                            lx1, ly1, lx2, ly2 = map(int, parts[2:6])
                            marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                            tile_mask = extract_mask_from_green(marked_tile)
                            offset = CONTEXT_PADDING // 2
                            logical_mask_part = tile_mask.crop((offset, offset, offset + (lx2 - lx1), offset + (ly2 - ly1)))
                            full_mask.paste(logical_mask_part, (lx1, ly1))
                        except: continue

            if not full_mask.getbbox(): continue

            mask_arr = np.array(full_mask)
            labeled_array, num_features = label(mask_arr > 128)
            print(f"  [{orig_file}] 发现 {num_features} 个潜在区域，生成聚焦切片...")
            
            # Coverage mask to prevent duplicate tiles for dense bars
            coverage_mask = np.zeros((h, w), dtype=bool)

            for i in range(1, num_features + 1):
                # Check if this island is already covered by previous tiles
                this_island_mask = (labeled_array == i)
                if np.all(coverage_mask[this_island_mask]):
                    continue

                cy, cx = center_of_mass(mask_arr, labeled_array, i)
                cx, cy = int(cx), int(cy)
                
                if self.args.mosaic == 1:
                    focus_size = int(h // 5 * 2)
                else :
                    focus_size = int(h // 4)

                half_size = focus_size // 2
                lx1 = max(0, cx - half_size)
                ly1 = max(0, cy - half_size)
                lx2 = min(w, lx1 + focus_size)
                ly2 = min(h, ly1 + focus_size)
                
                if (lx2 - lx1) < focus_size: lx1 = max(0, lx2 - focus_size)
                if (ly2 - ly1) < focus_size: ly1 = max(0, ly2 - focus_size)
                
                # Mark this tile area as covered
                coverage_mask[ly1:ly2, lx1:lx2] = True

                px1 = lx1 - CONTEXT_PADDING
                py1 = ly1 - CONTEXT_PADDING
                px2 = lx2 + CONTEXT_PADDING
                py2 = ly2 + CONTEXT_PADDING
                
                tile = extract_padded_tile(img, (px1, py1, px2, py2))
                tile_name = f"{idx}_T_{lx1}_{ly1}_{lx2}_{ly2}_P_{px1}_{py1}_{px2}_{py2}_focus{i}.png"
                tile.save(os.path.join(self.args.temp_tiles, tile_name))

    def run_hai_detection(self, mosaic_index, clear_existing=True):
        print(f"模式: HAI - 启动自动化检测 (Index: {mosaic_index})...")
        if clear_existing:
            if os.path.exists(self.hai_mask_folder): shutil.rmtree(self.hai_mask_folder)
            os.makedirs(self.hai_mask_folder)
        elif not os.path.exists(self.hai_mask_folder):
             os.makedirs(self.hai_mask_folder)

        try:
            self.update_hai_config(self.args.temp_tiles)
            app = Application(backend="uia").start(self.args.hai_path, wait_for_idle=False)
            time.sleep(15)
            desktop = Desktop(backend="uia")
            main_window = desktop.window(title_re=".*hentAI.*")
            main_window.wait('exists', timeout=20)
            main_window.set_focus()
            btns = main_window.descendants(control_type="Button")
            if len(btns) > mosaic_index:
                btns[mosaic_index].click_input()
                time.sleep(3)
            det_win = desktop.window(title_re=".*Detection.*")
            det_win.wait('exists', timeout=20)
            det_win.set_focus()
            f_btns = [b for b in det_win.descendants(control_type="Button") if b.window_text() not in ["最小化", "最大化", "关闭"]]
            if f_btns: f_btns[1].click_input()
            
            for _ in range(3600):
                try:
                    success = desktop.window(title_re=".*Success.*")
                    if success.exists(): success.close(); break
                except: pass
                time.sleep(1)
            if det_win.exists(): det_win.close()
            app.kill() 
        except Exception as e: print(f"GUI 自动化异常: {e}")

    def merge_detection_results(self):
        if not os.path.exists(self.hai_mask_folder): return
        print("模式: Merge - 正在合并所有检测切片为单一全图掩码...")
        
        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        for idx, orig_file in enumerate(input_files):
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            
            full_rough_mask = Image.new("L", (w, h), 0)
            tiles_to_delete = []
            
            for m_file in os.listdir(self.hai_mask_folder):
                if m_file.startswith(f"{idx}_T_") and m_file.endswith(".png"):
                    tiles_to_delete.append(os.path.join(self.hai_mask_folder, m_file))
                    parts = m_file.replace(".png", "").split("_")
                    try:
                        lx1, ly1, lx2, ly2 = map(int, parts[2:6])
                        marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                        tile_mask = extract_mask_from_green(marked_tile)
                        offset = CONTEXT_PADDING // 2
                        logical_mask_part = tile_mask.crop((offset, offset, offset + (lx2 - lx1), offset + (ly2 - ly1)))
                        full_rough_mask.paste(logical_mask_part, (lx1, ly1))
                    except: continue
            
            if not tiles_to_delete: continue

            p_box = (-CONTEXT_PADDING, -CONTEXT_PADDING, w + CONTEXT_PADDING, h + CONTEXT_PADDING)
            base_canvas = extract_padded_tile(original_img, p_box)
            
            green_canvas = Image.new("RGB", base_canvas.size, (0, 255, 0))
            final_mask_canvas = Image.new("L", base_canvas.size, 0)
            final_mask_canvas.paste(full_rough_mask, (CONTEXT_PADDING // 2, CONTEXT_PADDING // 2))
            
            merged_tile = Image.composite(green_canvas, base_canvas, final_mask_canvas)
            
            for p in tiles_to_delete:
                try: os.remove(p)
                except: pass
                
            save_path = os.path.join(self.hai_mask_folder, f"{idx}_T_0_0_{w}_{h}_merged.png")
            merged_tile.save(save_path)
            print(f"  [{orig_file}] 合并完成 -> {save_path}")

    def refine_masks_with_sam2_points(self, image_1024, initial_mask_1024, limit=25):
        """
        Refines mask using SAM 2. 
        Args:
            image_1024: A 1024x1024 PIL Image (or Tensor) representing the context.
            initial_mask_1024: A 1024x1024 PIL Image (L mode) containing the rough mask to refine.
        """
        mask_data = np.array(initial_mask_1024)
        refined_full_mask_acc = np.copy(mask_data)
        dark_limit = self.args.black_level
        # breakpoint()
        # --- PHASE 1: PREPARE PROMPTS FROM INITIAL MASK ---
        # We process input blocks to determine WHERE to click.
        labeled_array, num_features = label(mask_data > 128)
        if num_features == 0: 
            return initial_mask_1024
        # Sort input components by size just to process main parts first
        input_components = []
        for i in range(1, num_features + 1):
            area = np.sum(labeled_array == i)
            # Filter extremely small noise in input to avoid bad prompts
            if area > 10: 
                input_components.append((i, area))
        input_components.sort(key=lambda x: x[1], reverse=True)
        # Loop through input blobs to generate prompts
        for i, (label_id, area) in enumerate(input_components):
            
            # --- GENERATE POINT PROMPT (Centroid) ---
            coords = np.argwhere(labeled_array == label_id)
            median_idx = random.randint(0, len(coords)-1)
            center_y, center_x = coords[median_idx] # Prepare inputs for SAM
            img_gray = np.array(image_1024.convert("L"))
            current_try = 0
            while (img_gray[center_y,center_x]) > dark_limit and current_try < 50:
                median_idx =  random.randint(0, len(coords)-1)
                center_y, center_x = coords[median_idx] # Prepare inputs for SAM
                current_try += 1
            if current_try == 50 :
                continue
            # breakpoint()
            input_point = [[int(center_x), int(center_y)]]
            input_label = [1] # 1 = Foreground
        
            inputs = self.sam_processor(
                images=image_1024, 
                input_points=[[input_point]], 
                input_labels=[[input_label]], 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.sam_model(**inputs, multimask_output=False)
            
            pred_prob = torch.sigmoid(outputs.pred_masks).cpu().numpy().squeeze()
            pred_mask = (pred_prob > 0.5).astype(np.uint8) * 255
            pred_img = Image.fromarray(pred_mask).resize(image_1024.size, Image.NEAREST)

            pred_labeled, pred_num_features = label(pred_img)
            if pred_num_features == 0: continue
            sam_islands = []
            for j in range(1, pred_num_features + 1):
                island_area = np.sum(pred_labeled == j)
                sam_islands.append((j, island_area))
            # Sort SAM islands by size (Largest to Smallest)
            sam_islands.sort(key=lambda x: x[1], reverse=True)
            clean_island_mask = np.zeros_like(pred_img)
            if len(sam_islands) > 0:
                top_label, top_area = sam_islands[0]
                clean_island_mask[pred_labeled == top_label] = 1
                for j in range(1, len(sam_islands)):
                    curr_label, curr_area = sam_islands[j]
                    # mask_bool = ((curr_area) > 0)
                    masked_pixels = img_gray[pred_labeled == curr_label]
                    mean_val = np.mean(masked_pixels)
                    std_val = np.std(masked_pixels)
                    # print(mean_val,std_val)
                    # breakpoint()
                    if self.args.mosaic_color == 0: # BLACK
                        if not(mean_val < dark_limit and std_val < 15): continue
                    elif self.args.mosaic_color == 1: # WHITE
                        NotImplementedError
                    if curr_area < 8:
                        break
                    clean_island_mask[pred_labeled == curr_label] = 1
                    
            # if self.args.mosaic == 2: # Solid Bar
            clean_island_mask &= (img_gray < dark_limit)
            # Convert to uint8 for accumulation
            island_refined = (clean_island_mask > 0).astype(np.uint8) * 255
            # Merge into final result
            refined_full_mask_acc = np.maximum(refined_full_mask_acc, island_refined)
        
                
        # filt the mask again, remove the connection blocks that are too small (<5 pixels), the logics are same as above one 
        final_labeled, final_num_features = label(refined_full_mask_acc > 128)
        if final_num_features > 0:
            final_clean_mask = np.zeros_like(refined_full_mask_acc)
            for k in range(1, final_num_features + 1):
                block_area = np.sum(final_labeled == k)
                if block_area >= limit:  # Keep blocks 5 pixels or larger
                    final_clean_mask[final_labeled == k] = 255
            refined_full_mask_acc = final_clean_mask
        refined_full_mask_acc = Image.fromarray(refined_full_mask_acc)
        
        return refined_full_mask_acc
    
    def mask_refinement_process(self):
            self.load_sam2()
            if not os.path.exists(self.hai_mask_folder): return
            
            input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
            input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
            
            for idx, orig_file in enumerate(input_files):
                orig_path = os.path.join(self.args.input, orig_file)
                original_img = smart_resize(Image.open(orig_path).convert("RGB"))
                
                # if self.args.black_level > 0:
                original_img = self.preprocess_smart_detection(original_img)
                
                # CHANGE 1: Removed 'and "merged" not in f' to allow merged files
                tile_files = [f for f in os.listdir(self.hai_mask_folder) if f.startswith(f"{idx}_T_") and f.endswith(".png")]
                
                print(f"正在优化掩码: {orig_file} ({len(tile_files)} targets)")
                for m_file in tile_files:
                    # Initialize coordinates
                    px1, py1, px2, py2 = 0, 0, 0, 0
                    is_merged_file = "merged" in m_file

                    # CHANGE 2: Dual Regex logic
                    if is_merged_file:
                        match = re.search(r"T_(\d+)_(\d+)_(\d+)_(\d+)_merged", m_file)
                        if match:
                            px1, py1, px2, py2 = map(int, match.groups())
                        else:
                            print(f"Warning: Could not parse merged coords from {m_file}")
                            continue
                    else:
                        match = re.search(r"T_(\d+)_(\d+)_(\d+)_(\d+)_P_(-?\d+)_(-?\d+)_(\d+)_(\d+)", m_file)
                        if match:
                            px1, py1, px2, py2 = map(int, match.groups()[4:8])
                        else:
                            continue
                    
                    # Extract the clean partial image from source
                    clean_tile = extract_padded_tile(original_img, (px1, py1, px2, py2))
                    
                    file_path = os.path.join(self.hai_mask_folder, m_file)
                    marked_tile = Image.open(file_path).convert("RGB")
                        
                        # Ensure sizes match (vital for merged files if rounding errors occurred)
                    if clean_tile.size != marked_tile.size:
                        clean_tile = clean_tile.resize(marked_tile.size)

                    tile_mask = extract_mask_from_green(marked_tile)
                    if not tile_mask.getbbox(): continue

                        # Optional: Add specific debug stem for merged files
                    debug_stem = f"{idx}_merged" if is_merged_file else f"{idx}"
                        
                    # Run SAM2 Refinement
                    refined_mask = self.refine_masks_with_sam2_points(clean_tile, tile_mask) #, debug_stem=debug_stem) 
                        # Overwrite the file with new green mask
                    green_canvas = Image.new("RGB", clean_tile.size, (0, 255, 0))
                    refined_green_tile = Image.composite(green_canvas, clean_tile, refined_mask)
                
                    refined_green_tile.save(file_path)
                        
                    if is_merged_file:
                        print(f"  [Merged Refine] Finished: {m_file}")


def main():
    global LOGICAL_TILE_W, LOGICAL_TILE_H
    parser = argparse.ArgumentParser(description="Adaptive Buffer Manga Decensor")
    parser.add_argument("--mode", type=str, choices=["segment", "inpaint", "all", "hai", "refine", "extra"], default="all")
    parser.add_argument("--mosaic", type=int, default=1,help='1 for mosaic, 2 for bar')
    parser.add_argument("--mosaic_color", type=int, default=0,help='0 for black bars, 1 for white bars')

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--hai_path", type=str, default=DEFAULT_HAI_PATH)
    parser.add_argument("--temp_tiles", type=str, default=DEFAULT_TEMP_TILES_FOLDER)
    parser.add_argument("--black_level", type=int, default=20, help="0-255: Darkens gray pixels below this threshold to pure black to improve detection.")
    parser.add_argument("--min_size", type=int, default=1200, help="Upscale images smaller than this dimension")
    parser.add_argument("--sam_strategy", type=str, choices=["score", "area", "union", "min", "middle", "max"], default="score", help="Strategy to select SAM mask: 'score' (default), 'min' (smallest area), 'middle' (median), 'max' (largest), 'union' (merge all)")
    
    args = parser.parse_args()
    
    # Initialize Managers
    mask_manager = DecensorMaskManager(args)
    inpaint_manager = DecensorInpainter(args)
    
    # # Update global variable via arg
    global MIN_DIMENSION
    MIN_DIMENSION = args.min_size

    if args.mode == "segment": 
        mask_manager.segment_images_to_temp()
    elif args.mode == "hai": 
        mask_manager.run_hai_detection(args.mosaic)
    elif args.mode == "inpaint": 
        inpaint_manager.inpaint_process()
    elif args.mode == "refine": 
        mask_manager.mask_refinement_process()
    elif args.mode == "all":
        print(">>> 启动阶段 1: 全图粗略检测 (Coarse Detection) <<<")
        LOGICAL_TILE_W = 2000
        LOGICAL_TILE_H = 2000
        mask_manager.segment_images_to_temp()
        mask_manager.run_hai_detection(args.mosaic, clear_existing=True)
        # print(">>> 启动阶段 2: 重点区域二次检测 (Focused Detection) <<<")
        mask_manager.segment_focused_from_coarse()
        mask_manager.run_hai_detection(args.mosaic, clear_existing=False)
        if args.mosaic == 2:
            print(">>> 启动阶段 3: 切片掩码优化 (Refining Mask Tiles) <<<")
            mask_manager.mask_refinement_process()
            # print(">>> 启动阶段 3: 切片掩码再优化 (Refining Mask Tiles Twice) <<<")
        print(">>> 启动阶段 4: 合并检测结果 (Merging Masks) <<<")
        mask_manager.merge_detection_results()
        # if args.mosaic == 2:
            # mask_manager.mask_refinement_process()
        mask_manager.unload_sam2()
        cleanup_gpu()
        inpaint_manager.inpaint_process()


if __name__ == "__main__":
    main()
    
    
    

    # def refine_masks_with_sam2_points(self, image, initial_mask, debug_stem=""):
    #             mask_data = np.array(initial_mask)
    #             # Label connected components (islands)
    #             labeled_array, num_features = label(mask_data > 128)
    #             if num_features == 0: return initial_mask
                
    #             # Initialize with original mask to perform a UNION (Merge)
    #             refined_full_mask_acc = mask_data.copy()
                
    #             print(f" Detected {num_features} separate mask blocks. Processing (Box + Point Mode)...")

    #             for i in range(1, num_features + 1):
    #                 # Get all (y, x) coordinates for this island
    #                 coords = np.argwhere(labeled_array == i)
                    
    #                 # --- 1. GENERATE BOUNDING BOX (Simulating detection) ---
    #                 # Find min/max coordinates
    #                 y_min, x_min = np.min(coords, axis=0)
    #                 y_max, x_max = np.max(coords, axis=0)
                    
    #                 # Add Random Padding (as requested for testing)
    #                 # This simulates a "human" or "detector" providing a loose box
    #                 padx = np.random.rand() * (y_max-y_min)
    #                 pady = np.random.rand() * (x_max-x_min)

    #                 box = [
    #                     max(0, int(x_min - padx)), 
    #                     max(0, int(y_min - pady)), 
    #                     min(image.width, int(x_max + padx)), 
    #                     min(image.height, int(y_max + pady))
    #                 ]
                    
    #                 # --- 2. GENERATE POINTS (Interior) ---
    #                 points = []
    #                 labels = []
    #                 median_idx = len(coords) // 2
    #                 center_y, center_x = coords[median_idx] # Guaranteed interior point
    #                 geo_cy, geo_cx = np.mean(coords, axis=0)

    #                 if self.args.mosaic == 1:
    #                     # Strategy for Mosaics: Multi-point
    #                     points.append([int(center_x), int(center_y)])
    #                     labels.append(1)
                        
    #                     if len(coords) > 50: 
    #                         # Quadrant logic
    #                         q1 = coords[(coords[:,0] <= geo_cy) & (coords[:,1] <= geo_cx)]
    #                         q2 = coords[(coords[:,0] <= geo_cy) & (coords[:,1] > geo_cx)] 
    #                         q3 = coords[(coords[:,0] > geo_cy) & (coords[:,1] <= geo_cx)] 
    #                         q4 = coords[(coords[:,0] > geo_cy) & (coords[:,1] > geo_cx)] 
                            
    #                         for quad in [q1, q2, q3, q4]:
    #                             if len(quad) > 10: 
    #                                 q_idx = len(quad) // 2
    #                                 qy, qx = quad[q_idx]
    #                                 points.append([int(qx), int(qy)])
    #                                 labels.append(1)
    #                 else:
    #                     # Strategy for Black Bars
    #                     points.append([int(center_x), int(center_y)])
    #                     labels.append(1)
                    
    #                 MIN_SIDE_LENGTH = 256
    #                 w,h = image.size
    #                 w1 = box[2] - box[0]
    #                 h1 = box[3] - box[1]
    #                 scale_factor = 1.0
    #                 # Check if image is "Tiny" (e.g. < 512x512)
    #                 if w1 < MIN_SIDE_LENGTH or h1 < MIN_SIDE_LENGTH:
    #                     # Calculate scale to make the smallest side at least MIN_SIDE_LENGTH
    #                     scale_factor = max(MIN_SIDE_LENGTH / w1, MIN_SIDE_LENGTH / h1)
    #                     # Limit max scale to avoid memory explosion (e.g., max 8x)
    #                     scale_factor = min(scale_factor, 3)
    #                     new_w = int(w * scale_factor)
    #                     new_h = int(h * scale_factor)
    #                     # A. Resize Image
    #                     image1 = image.resize((new_w, new_h), Image.BICUBIC)
    #                     # B. Scale Box
    #                     box1 = [int(c * scale_factor) for c in box]
    #                     # C. Scale Points
    #                     points1 = [[int(p[0] * scale_factor), int(p[1] * scale_factor)] for p in points]
    #                     # print(f"   [Upscale] Scaling by {scale_factor:.2f}x to ({new_w}, {new_h}) for better SAM features.")
                    

    #                 # --- 4. RUN SAM 2 (With Points AND Boxes) ---
    #                 # Structure: 
    #                 # input_points: [ [ [x,y], [x,y] ] ] -> Batch=1, Object=1, Points=N
    #                 # input_boxes:  [ [ [x1,y1,x2,y2] ] ] -> Batch=1, Object=1, Box=1
                    
    #                 inputs = self.sam_processor(
    #                     images=image, 
    #                     input_points=[[points]], 
    #                     input_labels=[[labels]],
    #                     input_boxes=[[box]], # <--- ADDED BOX HERE
    #                     return_tensors="pt"
    #                 ).to(self.device)
                    
    #                 with torch.no_grad():
    #                     outputs = self.sam_model(**inputs)
                    
    #                 # Post-process
    #                 masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    #                 masks_np = masks[0].numpy() # Shape: (3, H, W)
    #                 scores_np = outputs.iou_scores.cpu().numpy()[0, 0] # Shape: (3,)
    #                 # =========================================================
    #                 # --- VISUALIZATION: SHOW RAW SAM OUTPUTS (ALL 3 MASKS) ---
    #                 # =========================================================
    #                 if debug_stem:
    #                     if not os.path.exists(DEFAULT_DEBUG_FOLDER): os.makedirs(DEFAULT_DEBUG_FOLDER)
                        
    #                     # 1. Base Image with Inputs (Box + Points)
    #                     vis_base = image.copy().convert("RGB")
    #                     draw = ImageDraw.Draw(vis_base)
    #                     draw.rectangle(box, outline="#00FFFF", width=4) # Cyan Box
    #                     for p in points:
    #                         r = 4
    #                         draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill="#FF0000", outline="white")

    #                     # 2. Create visualization for each of the 3 masks
    #                     mask_vis_list = []
    #                     for m_idx in range(3):
    #                         # Create a heatmap-style overlay for this mask
    #                         m_arr = (masks_np[m_idx] > 0).astype(np.uint8) * 255
    #                         m_pil = Image.fromarray(m_arr).convert("L")
                            
    #                         # Create red overlay
    #                         overlay = Image.new("RGB", image.size, (255, 0, 0))
    #                         # Composite: Original + Red Mask (alpha blended)
    #                         masked_comp = Image.composite(overlay, image.convert("RGB"), m_pil)
    #                         blended = Image.blend(image.convert("RGB"), masked_comp, 0.5)
                            
    #                         # Add Text: Score
    #                         d = ImageDraw.Draw(blended)
    #                         score_txt = f"M{m_idx}: {scores_np[m_idx]:.2f}"
    #                         d.text((10, 10), score_txt, fill="white")
                            
    #                         mask_vis_list.append(blended)

    #                     # 3. Stitch them together: [Input] [Mask0] [Mask1] [Mask2]
    #                     total_width = vis_base.width * 4
    #                     total_height = vis_base.height
                        
    #                     combo_img = Image.new('RGB', (total_width, total_height))
    #                     combo_img.paste(vis_base, (0, 0))
    #                     combo_img.paste(mask_vis_list[0], (vis_base.width, 0))
    #                     combo_img.paste(mask_vis_list[1], (vis_base.width * 2, 0))
    #                     combo_img.paste(mask_vis_list[2], (vis_base.width * 3, 0))
    #                     save_name = f"{debug_stem}_block{i}_SAM_ALL.jpg"
    #                     combo_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, save_name))
    #                     print(f"   [Debug] Saved raw SAM comparison: {save_name}")

    #                 # =========================================================
    #                 # --- CONTENT VALIDATION (Existing logic) ---
    #                 valid_indices = []
                    
    #                 if self.args.mosaic == 2: # Solid Bar
    #                     img_gray = np.array(image.convert("L"))
    #                     for m_idx in range(3):
    #                         mask_bool = masks_np[m_idx] > 0
    #                         if np.count_nonzero(mask_bool) == 0: continue
    #                         masked_pixels = img_gray[mask_bool]
    #                         mean_val = np.mean(masked_pixels)
    #                         std_val = np.std(masked_pixels)
                            
    #                         if self.args.mosaic_color == 0: # BLACK
    #                             if mean_val < 55 and std_val < 15.0:
    #                                 valid_indices.append(m_idx)
    #                         elif self.args.mosaic_color == 1: # WHITE
    #                             if mean_val > 200 and std_val < 15.0:
    #                                 valid_indices.append(m_idx)
    #                 else:
    #                     valid_indices = [0, 1, 2] # Mosaic accept all

    #                 if not valid_indices:
    #                     print(f"  [Filter] Block {i}: No valid SAM mask found. Keeping original.")
    #                     continue

    #                 # Select best mask
    #                 valid_masks_np = masks_np[valid_indices]
    #                 valid_areas = [np.sum(m > 0) for m in valid_masks_np]

    #                 best_relative_idx = 0 
    #                 if self.args.sam_strategy == "min":
    #                     best_relative_idx = np.argmin(valid_areas)
    #                 elif self.args.sam_strategy == "max" or self.args.sam_strategy == "area":
    #                     best_relative_idx = np.argmax(valid_areas)
    #                 original_idx = valid_indices[best_relative_idx]

    #                 island_refined = (masks_np[original_idx] > 0).astype(np.uint8) * 255

    #                 if scale_factor > 1.0:
    #                     high_res_pil = Image.fromarray(island_refined)
    #                     original_size_mask_pil = high_res_pil.resize((w, h), Image.NEAREST)
    #                     island_refined = np.array(original_size_mask_pil)
    #                 # Merge
    #                 refined_full_mask_acc = np.maximum(refined_full_mask_acc, island_refined)
                
    #             return Image.fromarray(refined_full_mask_acc)
        
    # def mask_refinement_process(self):
    #         self.load_sam2()
    #         if not os.path.exists(self.hai_mask_folder): return
            
    #         input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
    #         input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
            
    #         for idx, orig_file in enumerate(input_files):
    #             orig_path = os.path.join(self.args.input, orig_file)
    #             original_img = smart_resize(Image.open(orig_path).convert("RGB"))
                
    #             # if self.args.black_level > 0:
    #             original_img = self.preprocess_smart_detection(original_img)
                
    #             # CHANGE 1: Removed 'and "merged" not in f' to allow merged files
    #             tile_files = [f for f in os.listdir(self.hai_mask_folder) if f.startswith(f"{idx}_T_") and f.endswith(".png")]
                
    #             print(f"正在优化掩码: {orig_file} ({len(tile_files)} targets)")
    #             for m_file in tile_files:
    #                 # Initialize coordinates
    #                 px1, py1, px2, py2 = 0, 0, 0, 0
    #                 is_merged_file = "merged" in m_file

    #                 # CHANGE 2: Dual Regex logic
    #                 if is_merged_file:
    #                     match = re.search(r"T_(\d+)_(\d+)_(\d+)_(\d+)_merged", m_file)
    #                     if match:
    #                         px1, py1, px2, py2 = map(int, match.groups())
    #                     else:
    #                         print(f"Warning: Could not parse merged coords from {m_file}")
    #                         continue
    #                 else:
    #                     match = re.search(r"T_(\d+)_(\d+)_(\d+)_(\d+)_P_(-?\d+)_(-?\d+)_(\d+)_(\d+)", m_file)
    #                     if match:
    #                         px1, py1, px2, py2 = map(int, match.groups()[4:8])
    #                     else:
    #                         continue
                    
    #                 # Extract the clean partial image from source
    #                 clean_tile = extract_padded_tile(original_img, (px1, py1, px2, py2))
                    
    #                 file_path = os.path.join(self.hai_mask_folder, m_file)
    #                 marked_tile = Image.open(file_path).convert("RGB")
                        
    #                     # Ensure sizes match (vital for merged files if rounding errors occurred)
    #                 if clean_tile.size != marked_tile.size:
    #                     clean_tile = clean_tile.resize(marked_tile.size)

    #                 tile_mask = extract_mask_from_green(marked_tile)
    #                 if not tile_mask.getbbox(): continue

    #                     # Optional: Add specific debug stem for merged files
    #                 debug_stem = f"{idx}_merged" if is_merged_file else f"{idx}"
                        
    #                 # Run SAM2 Refinement
    #                 refined_mask = self.refine_masks_with_sam2_points(clean_tile, tile_mask) #, debug_stem=debug_stem) 
    #                     # Overwrite the file with new green mask
    #                 green_canvas = Image.new("RGB", clean_tile.size, (0, 255, 0))
    #                 refined_green_tile = Image.composite(green_canvas, clean_tile, refined_mask)
                
    #                 refined_green_tile.save(file_path)
                        
    #                 if is_merged_file:
    #                     print(f"  [Merged Refine] Finished: {m_file}")
