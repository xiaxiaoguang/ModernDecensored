import gradio as gr
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cv2
import random
from PIL import Image
import uuid
from scipy.ndimage import label, binary_dilation
import gc
# Make sure utils.tools is accessible in your environment as per your original code
from utils.tools import smart_resize 


# ================= CONFIGURATION =================
INPUT_DIR = "G://manga/K6"
OUTPUT_ROOT = "dataset_refined"

# Processing Settings
UI_RENDER_LIMIT = 1000
DATA_PROC_LIMIT = 2000      
CROP_SIZE = 1024
BRUSH_DEFAULT = 9

# ================= DIRECTORY SETUP =================
DIRS = {
    # Unified YOLO Data
    "yolo_img": os.path.join(OUTPUT_ROOT, "yolo", "images"),
    "yolo_lbl": os.path.join(OUTPUT_ROOT, "yolo", "labels"),
    
    # Dataset 1: Mosaic (High Texture)
    "mosaic_censored": os.path.join(OUTPUT_ROOT, "inpainter_mosaic", "censored"),
    "mosaic_mask":     os.path.join(OUTPUT_ROOT, "inpainter_mosaic", "mask"),
    "mosaic_gt":       os.path.join(OUTPUT_ROOT, "inpainter_mosaic", "ground_truth"),
    
    # Dataset 2: Black Bar (Flat Color)
    "bar_censored":    os.path.join(OUTPUT_ROOT, "inpainter_bar", "censored"),
    "bar_mask":        os.path.join(OUTPUT_ROOT, "inpainter_bar", "mask"),
    "bar_gt":          os.path.join(OUTPUT_ROOT, "inpainter_bar", "ground_truth"),
    
    # NEW: Anatomy Classification Labels
    "anatomy_lbl":     os.path.join(OUTPUT_ROOT, "anatomy_labels"),
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

def get_image_list():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if not os.path.exists(INPUT_DIR): return []
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    files.sort()
    return files

def get_scaled_image(img_path, target_size):
    """Resize maintaining aspect ratio."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = target_size / float(max(w, h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    print(f"Upscaling (Lanczos) to: {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)

def load_image(index):
    files = get_image_list()
    if not files:
        return None, "No images found.", 0, gr.update()
    
    index = max(0, min(int(index), len(files) - 1))
    img_path = os.path.join(INPUT_DIR, files[index])
    choices = [f"{i}: {name}" for i, name in enumerate(files)]
    
    try:
        ui_img = get_scaled_image(img_path, UI_RENDER_LIMIT)
        status = f"[{index + 1}/{len(files)}] {files[index]}"
        gc.collect()
        return ui_img, status, index, gr.update(choices=choices, value=choices[index])
    except Exception as e:
        return None, f"Error: {str(e)}", index, gr.update()

# --- UTILS ---
def apply_realistic_mosaic(img, mask, grid_size):
    """Area Averaging Mosaic."""
    if grid_size < 1: grid_size = 1
    y_idxs, x_idxs = np.where(mask)
    if len(y_idxs) == 0: return img

    y1, y2 = y_idxs.min(), y_idxs.max()
    x1, x2 = x_idxs.min(), x_idxs.max()
    
    roi = img[y1:y2+1, x1:x2+1]
    h, w = roi.shape[:2]
    
    small_h, small_w = max(1, h // grid_size), max(1, w // grid_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    local_mask = mask[y1:y2+1, x1:x2+1] > 0
    img[y1:y2+1, x1:x2+1][local_mask] = mosaic[local_mask]
    return img

def create_bar_geometry(mask_shape, single_bar_mask, style="Exact (Brush)"):
    final_mask = np.zeros(mask_shape, dtype=np.uint8)
    
    if style == "Smart Box":
        y_idxs, x_idxs = np.where(single_bar_mask)
        if len(y_idxs) == 0: return final_mask
        
        points = np.column_stack((x_idxs, y_idxs)).astype(np.int32)
        rect = cv2.minAreaRect(points)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(final_mask, [box], 0, 255, -1)
    else: 
        final_mask = (single_bar_mask > 0).astype(np.uint8) * 255

    return final_mask

# --- DYNAMIC GROUPING LOGIC ---
def find_optimal_grouping(mask_np, target_groups):
    target_groups = int(target_groups)
    if target_groups < 1: target_groups = 1 
    
    current_iter = 0
    max_iter = 150 
    step = 2 
    
    while current_iter < max_iter:
        if current_iter == 0:
            dilated = mask_np
        else:
            dilated = binary_dilation(mask_np, iterations=current_iter)
            
        labeled, num_found = label(dilated)
        
        if num_found <= target_groups:
            return labeled, num_found
        
        current_iter += step
        
    return labeled, num_found

# NEW: Added anatomy_class to arguments
def process_data(image_data, current_index, censor_type, bar_style, mosaic_size, target_group_count, anatomy_class):
    if image_data is None or 'background' not in image_data:
        return None, "No data.", current_index, gr.update()

    files = get_image_list()
    filename = files[current_index]
    file_stem = os.path.splitext(filename)[0]
    img_path = os.path.join(INPUT_DIR, filename)

    # 1. Load Data
    pil_clean = get_scaled_image(img_path, DATA_PROC_LIMIT)
    np_clean = np.array(pil_clean)
    h_img, w_img, _ = np_clean.shape
    
    # 2. Extract Mask
    if 'layers' in image_data and len(image_data['layers']) > 0:
        ui_mask_layer = image_data['layers'][0].convert("RGBA")
        ui_mask_np = (np.array(ui_mask_layer)[:, :, 3] > 0).astype(np.uint8)
        full_mask_pil = Image.fromarray(ui_mask_np * 255, mode='L').resize((w_img, h_img), Image.NEAREST)
        full_mask_np = (np.array(full_mask_pil) > 128).astype(np.uint8)
    else:
        return None, "Draw a mask first!", current_index, gr.update()

    # CORE LOGIC
    micro_labels, num_micros = label(full_mask_np)
    if num_micros == 0: return None, "No features found.", current_index, gr.update()

    bar_registry = {} 
    
    for i in range(1, num_micros + 1):
        single_stroke_mask = (micro_labels == i)
        if censor_type == "Black Bar":
            geom_mask = create_bar_geometry((h_img, w_img), single_stroke_mask, style=bar_style)
        else:
            geom_mask = (single_stroke_mask * 255).astype(np.uint8)
        bar_registry[i] = geom_mask

    if censor_type == "Black Bar":
        macro_labels, num_macros = find_optimal_grouping(full_mask_np, target_group_count)
    else:
        dilated_mask = binary_dilation(full_mask_np, iterations=5) 
        macro_labels, num_macros = label(dilated_mask)

    group_map = {g: [] for g in range(1, num_macros + 1)}
    for i in range(1, num_micros + 1):
        y, x = np.where(micro_labels == i)
        if len(y) > 0:
            g_id = macro_labels[y[0], x[0]]
            if g_id > 0:
                group_map[g_id].append(i)

    # YOLO GENERATION
    full_mask_combined = np.zeros((h_img, w_img), dtype=np.uint8)
    for b_mask in bar_registry.values():
        full_mask_combined = np.maximum(full_mask_combined, b_mask)
    
    np_censored_full = np_clean.copy()
    
    if censor_type == "Black Bar":
        np_censored_full[full_mask_combined > 0] = 0
    else:
        np_censored_full = apply_realistic_mosaic(np_censored_full, full_mask_combined, int(mosaic_size))

    yolo_labels = []
    class_id = 1 if censor_type == "Black Bar" else 0
    group_bboxes = {} 

    for g_id in range(1, num_macros + 1):
        group_specific_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for b_id in group_map[g_id]:
            group_specific_mask = np.maximum(group_specific_mask, bar_registry[b_id])
            
        y_idxs, x_idxs = np.where(group_specific_mask)
        if len(y_idxs) > 0:
            y1, y2, x1, x2 = y_idxs.min(), y_idxs.max(), x_idxs.min(), x_idxs.max()
            w, h = (x2-x1), (y2-y1)
            
            nx, ny = (x1 + w / 2) / w_img, (y1 + h / 2) / h_img
            nw, nh = w / w_img, h / h_img
            yolo_labels.append(f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            group_bboxes[g_id] = (x1, y1, x2, y2, x1 + w//2, y1 + h//2)

    # Save YOLO
    unique_id = uuid.uuid4().hex[:6]
    yolo_name = f"{file_stem}_{unique_id}"
    Image.fromarray(np_censored_full).save(os.path.join(DIRS["yolo_img"], f"{yolo_name}.jpg"), quality=95)
    with open(os.path.join(DIRS["yolo_lbl"], f"{yolo_name}.txt"), "w") as f:
        f.write("\n".join(yolo_labels))

    # ==========================================
    # NEW: Save Anatomy Classification Label
    # ==========================================
    with open(os.path.join(DIRS["anatomy_lbl"], f"{yolo_name}.txt"), "w") as f:
        f.write(anatomy_class)

    # INPAINTING GENERATION
    if censor_type == "Mosaic":
        target_dirs = (DIRS["mosaic_censored"], DIRS["mosaic_gt"], DIRS["mosaic_mask"])
        for g_id, bbox_data in group_bboxes.items():
            cx, cy = bbox_data[4], bbox_data[5]
            half = CROP_SIZE // 2
            x1 = max(0, min(w_img - CROP_SIZE, cx - half))
            y1 = max(0, min(h_img - CROP_SIZE, cy - half))
            x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE
            
            base = f"{yolo_name}_{g_id}"
            Image.fromarray(np_censored_full).crop((x1, y1, x2, y2)).save(os.path.join(target_dirs[0], f"{base}.png"))
            Image.fromarray(np_clean).crop((x1, y1, x2, y2)).save(os.path.join(target_dirs[1], f"{base}.png"))
            Image.fromarray(full_mask_combined).crop((x1, y1, x2, y2)).save(os.path.join(target_dirs[2], f"{base}.png"))

    else:
        target_dirs = (DIRS["bar_censored"], DIRS["bar_gt"], DIRS["bar_mask"])
        for g_id, bbox_data in group_bboxes.items():
            cx, cy = bbox_data[4], bbox_data[5]
            bars_in_group = group_map[g_id]
            
            half = CROP_SIZE // 2
            x1 = max(0, min(w_img - CROP_SIZE, cx - half))
            y1 = max(0, min(h_img - CROP_SIZE, cy - half))
            x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE
            
            context_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            for b_id, b_mask in bar_registry.items():
                if b_id not in bars_in_group:
                    b_crop = b_mask[y1:y2, x1:x2]
                    if np.any(b_crop):
                        context_mask = np.maximum(context_mask, b_crop)
            
            crop_clean = np_clean[y1:y2, x1:x2].copy()
            context_img = crop_clean.copy()
            context_img[context_mask > 0] = 0
            
            variations = []
            variations.append(bars_in_group)
            
            for v_idx, subset in enumerate(variations):
                var_img = context_img.copy()
                var_total_mask = context_mask.copy()
                
                for b_id in subset:
                    b_crop = bar_registry[b_id][y1:y2, x1:x2]
                    if np.any(b_crop):
                        var_img[b_crop > 0] = 0
                        var_total_mask = np.maximum(var_total_mask, b_crop)
                
                base = f"{yolo_name}_{g_id}_{v_idx}"
                Image.fromarray(var_img).save(os.path.join(target_dirs[0], f"{base}.png"))
                Image.fromarray(crop_clean).save(os.path.join(target_dirs[1], f"{base}.png"))
                Image.fromarray(var_total_mask).save(os.path.join(target_dirs[2], f"{base}.png"))

    del np_clean, np_censored_full
    gc.collect()

    return load_image(current_index + 1)

def jump_to_image(choice):
    if not choice: return None, "Select an image", 0, gr.update()
    index = int(choice.split(":")[0])
    return load_image(index)

# ================= UI LAYOUT =================
with gr.Blocks(title="Micro-Bar Labeler", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ⚡ Micro-Bar Labeler")
    
    current_index = gr.State(value=0)
    
    with gr.Row():
        with gr.Column(scale=4):
            editor = gr.ImageEditor(
                label="Workspace", type="pil", interactive=True,
                brush=gr.Brush(colors=["#00FF00"], default_size=BRUSH_DEFAULT), 
                height=800 
            )
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False)
            image_selector = gr.Dropdown(label="Jump to Image", choices=[])
            
            gr.Markdown("### 🛠️ Settings")
            censor_mode = gr.Radio(["Mosaic", "Black Bar"], label="Type", value="Black Bar")
            
            # NEW: UI for Anatomy Classification
            with gr.Group():
                gr.Markdown("**Anatomy Classification**")
                anatomy_class = gr.Radio(
                    ["Penis", "Vagina", "Penis+Vagina", "Other"], 
                    label="Subject", 
                    value="Penis"
                )
            
            with gr.Group():
                gr.Markdown("**Black Bar Grouping**")
                target_groups = gr.Number(value=1, label="Target Groups", precision=0, 
                                          info="Auto-dilates to find this many clusters.")
                bar_style = gr.Radio(["Exact (Brush)", "Smart Box"], value="Exact (Brush)", label="Shape")

            with gr.Group():
                gr.Markdown("**Mosaic**")
                mosaic_scale = gr.Slider(4, 30, 7, step=1, label="Grid Size")

            save_btn = gr.Button("💾 PROCESS", variant="primary", size="lg")
            
            with gr.Row():
                prev_btn = gr.Button("⬅️")
                skip_btn = gr.Button("➡️")

    demo.load(load_image, inputs=[current_index], outputs=[editor, status, current_index, image_selector])
    image_selector.change(jump_to_image, inputs=[image_selector], outputs=[editor, status, current_index, image_selector])
    
    # NEW: Added anatomy_class to inputs array
    save_btn.click(process_data, 
                   inputs=[editor, current_index, censor_mode, bar_style, mosaic_scale, target_groups, anatomy_class], 
                   outputs=[editor, status, current_index, image_selector])
    
    prev_btn.click(lambda idx: load_image(idx - 1), inputs=[current_index], outputs=[editor, status, current_index, image_selector])
    skip_btn.click(lambda idx: load_image(idx + 1), inputs=[current_index], outputs=[editor, status, current_index, image_selector])

if __name__ == "__main__":
    demo.launch()