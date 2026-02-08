import gradio as gr
import os
import re
import numpy as np
from PIL import Image, ImageDraw

# ================= CONFIGURATION =================
INPUT_FOLDER = r"G:\dataset\input"          # Your raw images
HAI_MASK_FOLDER = r"G:\dataset\hai_masks"   # Where the script looks for _merged.png
# =================================================

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_file_list():
    if not os.path.exists(INPUT_FOLDER): return []
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
    files.sort(key=natural_sort_key)
    return files

def process_drawing(file_name, editor_data, padding):
    """
    Calculates the bbox from the user's drawing and generates the preview.
    """
    if not file_name or editor_data is None:
        return None, "Please select an image and draw.", None

    # Handle Gradio 4.x ImageEditor output
    # 'composite' usually contains the full image with drawing
    # 'layers' contains just the drawing (if supported), but composite is safer fallback
    if isinstance(editor_data, dict):
        original = editor_data.get("background")
        # We need the mask. In Gradio 4, 'layers' is a list of RGBA images. 
        # The user draws on the first layer.
        layers = editor_data.get("layers", [])
        
        if layers:
            mask_pil = layers[0]
            # Extract alpha or checking for non-transparent pixels
            mask_arr = np.array(mask_pil)
            # Check Alpha channel (index 3) > 0
            if mask_arr.shape[2] == 4:
                is_drawn = mask_arr[:, :, 3] > 0
            else:
                # Fallback if no alpha (unlikely for layers)
                is_drawn = np.any(mask_arr > 0, axis=2)
        else:
            # Fallback for older gradio or simple composite
            composite = editor_data.get("composite")
            orig_arr = np.array(original)
            comp_arr = np.array(composite)
            diff = np.abs(orig_arr.astype(int) - comp_arr.astype(int))
            is_drawn = np.sum(diff, axis=2) > 10 # Tolerance
    else:
        return None, "Error: Invalid data format", None

    if not np.any(is_drawn):
        return None, "No drawing detected. Draw on the image first.", None

    # 1. Calculate Bounding Box of the Scribble
    rows = np.any(is_drawn, axis=1)
    cols = np.any(is_drawn, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 2. Apply Padding
    w, h = original.size
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w, x_max + padding)
    y2 = min(h, y_max + padding)

    # 3. Create Preview Image (Original + Red Box)
    preview_img = original.copy()
    draw = ImageDraw.Draw(preview_img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    # Return data needed for saving
    save_info = {
        "bbox": (x1, y1, x2, y2),
        "original": original,
        "mask_layer": mask_pil if 'mask_pil' in locals() else None,
        "file_name": file_name
    }
    
    return preview_img, f"BBox: {x1},{y1} - {x2},{y2}", save_info

def save_merged_file(save_info):
    if not save_info:
        return "Nothing to save. Please draw and verify first."

    x1, y1, x2, y2 = save_info["bbox"]
    original = save_info["original"]
    mask_layer = save_info["mask_layer"]
    filename = save_info["file_name"]

    # 1. Crop the Original
    crop_img = original.crop((x1, y1, x2, y2))
    
    # 2. Crop the Mask (User's scribble)
    if mask_layer:
        mask_crop = mask_layer.crop((x1, y1, x2, y2))
    else:
        return "Error: Could not retrieve mask layer."

    # 3. Composite: We need the crop + Green Mask on top
    # The pipeline expects Green pixels to indicate "Target".
    # We will overlay the user's green scribble onto the crop.
    
    # Ensure mask is green (Gradio brush might be any color, we force green)
    mask_arr = np.array(mask_crop)
    # Create a pure green image
    green_solid = np.zeros_like(mask_arr)
    green_solid[:] = [0, 255, 0, 255] # Green, Full Alpha
    
    # Where the user drew (alpha > 0), use the Green Solid
    alpha = mask_arr[:, :, 3]
    user_drawn_mask = alpha > 0
    
    final_arr = np.array(crop_img.convert("RGBA"))
    # Overlay green
    final_arr[user_drawn_mask] = [0, 255, 0, 255]
    
    final_output = Image.fromarray(final_arr).convert("RGB")

    # 4. Generate Filename
    # Get Index
    all_files = get_file_list()
    try:
        idx = all_files.index(filename)
    except:
        idx = 0 # Fallback

    # Filename format: {idx}_T_{x1}_{y1}_{x2}_{y2}_merged.png
    out_name = f"{idx}_T_{x1}_{y1}_{x2}_{y2}_merged.png"
    
    if not os.path.exists(HAI_MASK_FOLDER):
        os.makedirs(HAI_MASK_FOLDER)
    
    out_path = os.path.join(HAI_MASK_FOLDER, out_name)
    final_output.save(out_path)

    return f"Saved: {out_name}\nNow run your refinement script!"

# ================= UI LAYOUT =================
with gr.Blocks(title="SAM2 BBox debugger") as app:
    state_store = gr.State() # To hold image data between preview and save
    
    gr.Markdown("## SAM2 BBox Manual Injector")
    gr.Markdown("Draw on the target. The red box shows the crop area sent to SAM.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_dd = gr.Dropdown(label="Files", choices=get_file_list())
            padding_slider = gr.Slider(label="BBox Padding (px)", minimum=0, maximum=100, value=20, step=5)
            refresh_btn = gr.Button("Refresh Files")
            
        with gr.Column(scale=3):
            # Editor
            editor = gr.ImageEditor(
                label="Draw Green Scribble",
                type="pil",
                brush=gr.Brush(colors=["#00FF00"], default_size=15),
                interactive=True,
                height=600
            )
            
            with gr.Row():
                verify_btn = gr.Button("1. Verify BBox", variant="secondary")
                save_btn = gr.Button("2. Save for Pipeline", variant="primary")
            
            status_txt = gr.Textbox(label="Status")
            
    # Logic
    refresh_btn.click(lambda: gr.update(choices=get_file_list()), outputs=[file_dd])
    
    file_dd.change(lambda x: Image.open(os.path.join(INPUT_FOLDER, x)).convert("RGB"), inputs=[file_dd], outputs=[editor])

    # Verify Click
    verify_btn.click(
        fn=process_drawing,
        inputs=[file_dd, editor, padding_slider],
        outputs=[editor, status_txt, state_store] 
        # Note: We update 'editor' background to show the Red Box preview
    )

    # Save Click
    save_btn.click(
        fn=save_merged_file,
        inputs=[state_store],
        outputs=[status_txt]
    )

if __name__ == "__main__":
    app.launch()