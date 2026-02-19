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
from scipy.ndimage import label, binary_dilation, center_of_mass 
from diffusers import DiffusionPipeline
from transformers import Sam2Processor, Sam2Model
from utils.inpainter import DecensorInpainter,DecensorInpainter2
from utils.inpainter2 import DecensorInpainterX
from utils.tools import *
import cv2 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
        Targeted Preprocessing: Heals dents and holes in existing black bars caused by 
        overlapping manga effects (liquids, bubbles) using Convex Hulls, 
        without hallucinating new bars.
        """
        import cv2
        import numpy as np
        from PIL import Image

        img_np = np.array(img)
        
        # 1. Working Channel: 'L' from LAB is best for lightness separation
        if img_np.ndim == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
        else:
            l_channel = img_np.copy()

        # --- Fast Local Variance Map ---
        k_size = (5, 5) 
        mean_l = cv2.boxFilter(l_channel, cv2.CV_32F, k_size)
        sqr_l = cv2.sqrBoxFilter(l_channel, cv2.CV_32F, k_size)
        variance = np.abs(sqr_l - mean_l**2)
        std_dev = np.sqrt(variance)

        # --- Base Mask Creation ---
        intensity_limit = self.args.black_level if self.args.black_level > 0 else 30
        flatness_limit = 5.0  
        
        is_dark = l_channel < intensity_limit
        is_flat = std_dev < flatness_limit
        
        raw_mask = np.bitwise_and(is_dark, is_flat.astype(bool)).astype(np.uint8) * 255

        # Light cleanup to remove tiny stray pixels before contouring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

        # --- STRATEGY: CONVEX HULL HEALING ---
        # Find the outlines of existing, continuous black regions
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        repaired_mask = np.zeros_like(clean_mask)
        
        # Filter out random small screentone dust so we only process sizable chunks
        MIN_AREA_THRESHOLD = 150 
        
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA_THRESHOLD:
                # The Convex Hull snaps a tight boundary around the outermost points of the blob.
                # This naturally bridges any "bites" or "dents" taken out by overlapping effects.
                hull = cv2.convexHull(cnt)
                
                # Draw the repaired shape completely filled in to crush internal text/noise
                cv2.drawContours(repaired_mask, [hull], 0, 255, thickness=cv2.FILLED)

        # A mild morphological close just to smooth the repaired edges slightly
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_mask = cv2.morphologyEx(repaired_mask, cv2.MORPH_CLOSE, smooth_kernel)

        # --- RENDERING (CLAHE + Black Crush) ---
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_channel)
        # Image.fromarray(final_mask).save("mask.png")
        # Apply our tightly healed mask back to the image
        l_enhanced[final_mask > 0] = 0
        
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

    def run_yolo_detection(self):
        """
        Replaces tiling, GUI automation, and merging by running YOLO natively
        on the full image, extracting pure black pixels inside the bounding boxes,
        filtering out noise by keeping only the largest connected block,
        and rendering them as a green mask over the original image.
        """
        from ultralytics import YOLO

        print("模式: YOLO - 启动端到端全图检测与掩码生成...")
        
        if os.path.exists(self.hai_mask_folder): shutil.rmtree(self.hai_mask_folder)
        os.makedirs(self.hai_mask_folder)

        print(f"Loading YOLO weights from {YOLO_PATH}...")
        model = YOLO(YOLO_PATH)
        
        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        for idx, orig_file in enumerate(input_files):
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            original_img = self.preprocess_smart_detection(original_img)
            # original_img.save("1.png")
            # breakpoint()
            w, h = original_img.size
            
            # Predict directly on the high-res original image
            results = model.predict(orig_path, imgsz=1024, conf=0.05, iou=0.6, device=self.device, verbose=False)
            # Convert PIL image to OpenCV BGR for fast NumPy slicing
            orig_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
            # Create a blank grayscale canvas for the precise black pixel mask
            full_rough_mask = np.zeros((h, w), dtype=np.uint8)
            
            found_bars = 0
            for box in results[0].boxes:
                if int(box.cls[0]) == 1: # Assuming Class 1 is 'black_bar'
                    found_bars += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Crop the region of interest defined by the bounding box
                    roi = orig_cv[y1:y2, x1:x2]
                    # Find pure black pixels 
                    is_black = np.all(roi <= [5, 5, 5], axis=-1)
                    # Convert boolean mask to uint8 for OpenCV processing
                    roi_mask_uint8 = (is_black * 255).astype(np.uint8)
                    # --- NEW FILTERING LOGIC ---
                    # Find all connected components in this bounding box
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask_uint8, connectivity=8)
                    
                    MAX_COMPONENTS = 3
                    MIN_AREA_RATIO = 0.1
                    if num_labels > 1:
                        # Extract areas of all foreground components (ignore background at index 0)
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        
                        # Sort indices by area in descending order
                        sorted_indices = np.argsort(areas)[::-1]
                        
                        # Find the area of the absolute largest component
                        largest_area = areas[sorted_indices[0]]
                        valid_indices = []
                        for idx2 in sorted_indices[:MAX_COMPONENTS]:
                            if areas[idx2] >= largest_area * MIN_AREA_RATIO:
                                # Add 1 to shift back to actual label IDs (since we sliced [1:] earlier)
                                valid_indices.append(idx2 + 1)
                                
                        clean_is_black = np.isin(labels, valid_indices)
                        full_rough_mask[y1:y2, x1:x2][clean_is_black] = 255
            
            if MASK_EXPANSION_PIXELS > 0:
                k_size = 2 * MASK_EXPANSION_PIXELS + 1
                expansion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                full_rough_mask = cv2.dilate(full_rough_mask, expansion_kernel, iterations=1)
            
            full_rough_mask_pil = Image.fromarray(full_rough_mask)
            
            # --- Padding/composite logic ---
            p_box = (-CONTEXT_PADDING, -CONTEXT_PADDING, w + CONTEXT_PADDING, h + CONTEXT_PADDING)
            base_canvas = extract_padded_tile(original_img, p_box)
            
            green_canvas = Image.new("RGB", base_canvas.size, (0, 255, 0))
            final_mask_canvas = Image.new("L", base_canvas.size, 0)
            
            # Paste the extracted black bars into the padded canvas
            final_mask_canvas.paste(full_rough_mask_pil, (CONTEXT_PADDING // 2, CONTEXT_PADDING // 2))
            
            # Composite the bright green pixels over the base image using the precise mask
            merged_tile = Image.composite(green_canvas, base_canvas, final_mask_canvas)
            
            # Save exactly as the downstream SAM2 logic expects
            save_path = os.path.join(self.hai_mask_folder, f"{idx}_T_0_0_{w}_{h}_merged.png")
            merged_tile.save(save_path)
            print(f"  [{orig_file}] YOLO 检测完成 -> 发现 {found_bars} 个黑条框 -> {save_path}")
            
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
                # original_img = self.preprocess_smart_detection(original_img)
                
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
    parser.add_argument("--mosaic", type=int, default=2,help='1 for mosaic, 2 for bar')
    parser.add_argument("--mosaic_color", type=int, default=0,help='0 for black bars, 1 for white bars')

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--hai_path", type=str, default=DEFAULT_HAI_PATH)
    parser.add_argument("--temp_tiles", type=str, default=DEFAULT_TEMP_TILES_FOLDER)
    parser.add_argument("--black_level", type=int, default=30, help="0-255: Darkens gray pixels below this threshold to pure black to improve detection.")
    parser.add_argument("--min_size", type=int, default=1200, help="Upscale images smaller than this dimension")
    parser.add_argument("--sam_strategy", type=str, choices=["score", "area", "union", "min", "middle", "max"], default="score", help="Strategy to select SAM mask: 'score' (default), 'min' (smallest area), 'middle' (median), 'max' (largest), 'union' (merge all)")
    
    args = parser.parse_args()
    
    # Initialize Managers
    mask_manager = DecensorMaskManager(args)
    inpaint_manager = DecensorInpainter(args)
    
    # # Update global variable via arg
    global MIN_DIMENSION
    MIN_DIMENSION = args.min_size

    if args.mode == "hai": 
        mask_manager.run_yolo_detection()
    elif args.mode == "inpaint": 
        inpaint_manager.inpaint_process()
    elif args.mode == "refine": 
        mask_manager.mask_refinement_process()
    elif args.mode == "all":
        mask_manager.run_yolo_detection()
        mask_manager.mask_refinement_process()
        mask_manager.unload_sam2()
        inpaint_manager.inpaint_process()

if __name__ == "__main__":
    main()
    