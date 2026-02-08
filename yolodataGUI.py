import gradio as gr
import numpy as np
import os
import cv2
from PIL import Image
import uuid
from scipy.ndimage import label, binary_dilation
import gc

# ================= CONFIGURATION =================
INPUT_DIR = "G://manga/K5"
OUTPUT_DIR = "dataset_yolo"  # Changed folder name to indicate structure
UI_RENDER_LIMIT = 1000       # Resolution for Browser (Fast)
DATA_PROC_LIMIT = 2000       # Resolution for Saving (High Quality)
BRUSH_SIZE = 10              # Thicker default brush
# =================================================

# Create YOLO directory structure
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

def get_image_list():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if not os.path.exists(INPUT_DIR): return []
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    files.sort()
    return files

def get_scaled_image(img_path, target_size):
    """Utility to resize image while maintaining aspect ratio."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    # Only downscale if the image is massive
    if max(w, h) > target_size:
        scale = target_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

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

def process_and_save_yolo(image_data, current_index, censor_mode):
    if image_data is None or 'background' not in image_data:
        return None, "No data.", current_index, gr.update()

    files = get_image_list()
    img_path = os.path.join(INPUT_DIR, files[current_index])
    original_filename = files[current_index]
    
    # 1. Load High-Res Image (The YOLO input)
    data_bg = get_scaled_image(img_path, DATA_PROC_LIMIT)
    img_w, img_h = data_bg.size
    bg_np = np.array(data_bg)
    
    # 2. Extract Mask & Rescale
    if 'layers' in image_data and len(image_data['layers']) > 0:
        ui_mask_layer = image_data['layers'][0].convert("RGBA")
        ui_mask_np = (np.array(ui_mask_layer)[:, :, 3] > 0).astype(np.uint8)
    else:
        return None, "Draw a mask first!", current_index, gr.update()

    full_mask_pil = Image.fromarray(ui_mask_np * 255, mode='L').resize((img_w, img_h), Image.NEAREST)
    full_mask_np = (np.array(full_mask_pil) > 128).astype(np.uint8)

    # 3. Detect Individual Objects
    structure = np.ones((3, 3), dtype=int)
    # Dilate slightly to connect messy brush strokes
    dilated_mask = binary_dilation(full_mask_np, iterations=5)
    labeled_array, num_features = label(dilated_mask, structure=structure)

    if num_features == 0:
        return None, "No objects detected.", current_index, gr.update()

    yolo_labels = [] # To store "class x y w h" strings

    # Map Mode to Class ID
    # 0 = Mosaic, 1 = Bar
    class_id = 1 if censor_mode == "Black Bar" else 0

    # 4. Iterate over ALL detected blobs
    for i in range(1, num_features + 1):
        # Create mask for JUST this object
        obj_mask = (labeled_array == i).astype(np.uint8) * 255
        
        # --- A. VISUAL GENERATION (Modify the image) ---
        if censor_mode == "Black Bar":
            # Smart Rotated Bar Logic
            y_idxs, x_idxs = np.where(obj_mask)
            points = np.column_stack((x_idxs, y_idxs)).astype(np.int32)
            rect = cv2.minAreaRect(points)
            box = np.int32(cv2.boxPoints(rect))
            
            # Draw the rotated bar visually on the image
            cv2.drawContours(bg_np, [box], 0, (0, 0, 0), -1)
            
            # For YOLO Label: Get the Axis-Aligned Bounding Box of the rotated bar
            # (Because standard YOLO detects upright boxes)
            x, y, w, h = cv2.boundingRect(box)
            
        else:
            # Mosaic Logic
            y_idxs, x_idxs = np.where(obj_mask)
            y1, y2 = y_idxs.min(), y_idxs.max()
            x1, x2 = x_idxs.min(), x_idxs.max()
            
            # Extract and pixelate
            cell = bg_np[y1:y2+1, x1:x2+1]
            if cell.size == 0: continue
            
            ratio = 10 # Strong mosaic
            small = cv2.resize(cell, (max(1, cell.shape[1]//ratio), max(1, cell.shape[0]//ratio)), interpolation=cv2.INTER_NEAREST)
            mosaiced = cv2.resize(small, (cell.shape[1], cell.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Apply back only where the mask is
            local_mask = (obj_mask[y1:y2+1, x1:x2+1] > 0)
            bg_np[y1:y2+1, x1:x2+1][local_mask] = mosaiced[local_mask]
            
            # Label Coordinates
            x, y, w, h = x1, y1, (x2-x1), (y2-y1)

        # --- B. LABEL GENERATION (Normalize) ---
        # Ensure box is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Convert Top-Left (x,y) to Center (cx, cy)
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        
        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # 5. Save FULL Image and ONE Label File
    # Generate unique filename based on original to prevent overwrite collision if needed
    # or just use original name
    out_name = os.path.splitext(original_filename)[0] + f"_{uuid.uuid4().hex[:4]}"
    
    # Save Image
    final_img = Image.fromarray(bg_np)
    final_img.save(os.path.join(OUTPUT_DIR, "images", f"{out_name}.jpg"), quality=95)
    
    # Save Label
    with open(os.path.join(OUTPUT_DIR, "labels", f"{out_name}.txt"), "w") as f:
        f.write("\n".join(yolo_labels))

    # Cleanup
    del data_bg
    gc.collect()

    return load_image(current_index + 1)

def jump_to_image(choice):
    if not choice: return None, "Select an image", 0, gr.update()
    index = int(choice.split(":")[0])
    return load_image(index)

# --- UI LAYOUT ---
with gr.Blocks(title="YOLO Finetune Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🚀 YOLO Dataset Generator (Full Page)")
    gr.Markdown("Draw ALL censored areas on the page. Saves `images/name.jpg` and `labels/name.txt`.")
    
    current_index = gr.State(value=0)
    
    with gr.Row():
        with gr.Column(scale=4):
            editor = gr.ImageEditor(
                label="Workspace", type="pil", interactive=True,
                # THICKER BRUSH DEFAULT
                brush=gr.Brush(colors=["#00FF00"], default_size=BRUSH_SIZE), 
                height=800 
            )
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False)
            image_selector = gr.Dropdown(label="Jump to Index", choices=[])
            
            censor_mode = gr.Radio(
                ["Mosaic", "Black Bar"], 
                label="Class Type", 
                value="Black Bar",
                info="Mosaic = Class 0\nBlack Bar = Class 1"
            )

            save_btn = gr.Button("💾 SAVE YOLO DATA", variant="primary", size="lg")
            
            with gr.Row():
                prev_btn = gr.Button("⬅️ Prev")
                skip_btn = gr.Button("Next ➡️")
    
    # Logic
    demo.load(load_image, inputs=[current_index], outputs=[editor, status, current_index, image_selector])
    image_selector.change(jump_to_image, inputs=[image_selector], outputs=[editor, status, current_index, image_selector])
    
    save_btn.click(process_and_save_yolo, 
                   inputs=[editor, current_index, censor_mode], 
                   outputs=[editor, status, current_index, image_selector])
    
    prev_btn.click(lambda idx: load_image(idx - 1), inputs=[current_index], outputs=[editor, status, current_index, image_selector])
    skip_btn.click(lambda idx: load_image(idx + 1), inputs=[current_index], outputs=[editor, status, current_index, image_selector])

if __name__ == "__main__":
    demo.launch()