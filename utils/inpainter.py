import os
import time
import torch
import gc
import numpy as np
import configparser
import shutil
import argparse
import re

from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageEnhance
from scipy.ndimage import label, binary_dilation, center_of_mass, find_objects
from peft import PeftModel

# --- IMPORTS FOR LOADING ---
from huggingface_hub import hf_hub_download
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, AutoPipelineForInpainting, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from transformers import Sam2Processor, Sam2Model
import cv2
# Placeholder for local tools if they exist, otherwise ignore
from .tools import *

# Define your LoRA path (update this to your output folder)

class DecensorInpainter:
    def __init__(self, args):
        self.args = args
        self.hai_dir = os.path.dirname(args.hai_path) if hasattr(args, 'hai_path') else "hai_output"
        self.hai_mask_folder = os.path.join(self.hai_dir, "decensor_input")
        self.pipeline = None
        self.target_resolution = 1024 
        self.lora_root = LORA_PATH # Assuming Windows path structure from your snippet
        self.is_loaded = False

    def load_pipeline(self):
        """
        Initializes the model, loads the SPECIFIC LoRA needed, and fuses it.
        """
        if self.is_loaded: 
            return

        print(f"[Init] Loading Base Model: ShinoharaHare/Waifu-Inpaint-XL")
        self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "ShinoharaHare/Waifu-Inpaint-XL",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        target_lora_name = "mosaic" if self.args.mosaic == 1 else "bar"
        lora_path = os.path.join(self.lora_root, target_lora_name)
        self.pipeline.vae.to(torch.float32)
        if os.path.exists(lora_path):
            print(f"[Init] Loading LoRA from: {lora_path}")
            try:
                self.pipeline.load_lora_weights(lora_path,weight_name="pytorch_lora_weights.safetensors")
                self.pipeline.fuse_lora(lora_scale=1)
                print(f"[Init] LoRA '{target_lora_name}' successfully fused.")
            except Exception as e:
                print(f"[Error] Failed to load/fuse LoRA: {e}")
        else:
            print(f"[Init] LoRA path not found ({lora_path}). Using Base Model.")

        self.is_loaded = True
            
    def preprocess_smart_detection(self, img):
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
    
    def get_prompts(self):
        # Prompts optimized for Waifu-Inpaint-XL
        return {
            "prompt": "reconstruct, genitals detail, high quality, uncensored, lineart",
            "negative_prompt": "mosaic, black bars, low quality, error"
        }

    def save_debug_images(self, stem, feat_id, tile_img, mask_img, inpainted_tile=None):
        if not os.path.exists(DEFAULT_DEBUG_FOLDER): os.makedirs(DEFAULT_DEBUG_FOLDER)
        base_name = f"{stem}_F{feat_id}"
        tile_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_0_raw.png"))
        mask_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_1_mask.png"))
        if inpainted_tile:
            inpainted_tile.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_3_inpainted.png"))
            
    def inpaint_process(self):
        self.load_pipeline()
        
        if not os.path.exists(self.args.output): os.makedirs(self.args.output)
        
        adaptive_strength = 0.66 if self.args.mosaic == 1 else 0.99
        num_inference_steps = 40 if self.args.mosaic == 1 else 35
        prompts = self.get_prompts()

        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])


        for idx, orig_file in enumerate(input_files):
            print(f"Processing: {orig_file}")
            orig_path = os.path.join(self.args.input, orig_file)
            
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            # if self.args.black_level > 0:
            original_img = self.preprocess_smart_detection(original_img)

            # --- MASK LOADING ---
            full_mask = Image.new("L", (w, h), 0)
            prefix = f"{idx}_" 
            
            if os.path.exists(self.hai_mask_folder):
                for m_file in os.listdir(self.hai_mask_folder):
                    if m_file.startswith(prefix) and "_T_" in m_file and m_file.endswith(".png"):
                        parts = m_file.replace(".png", "").split("_")
                        try:
                            l_x1, l_y1, l_x2, l_y2 = map(int, parts[2:6])
                            marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                            tile_mask = extract_mask_from_green(marked_tile)
                            full_mask.paste(tile_mask, (l_x1, l_y1))
                        except Exception: continue
            
            if not full_mask.getbbox():
                original_img.save(os.path.join(self.args.output, f"{idx}.png"))
                continue

            output_canvas = np.array(original_img).astype(np.float32)
            mask_data = np.array(full_mask)

            # --- UPDATED CLUSTERING LOGIC (Auto-Dilation) ---
            # Strategy: Fuse disjoint masks into coherent islands
            
            final_dilation_iter = 30 # Default baseline
            if self.args.mosaic == 2:
                print("  [Auto-Group] Analyzing sparse connectivity...")
                # Search range: 20px to 140px, step 20px
                search_steps = range(30, 151, 20)
                found_optimal = False
                for step_d in search_steps:
                    # Test current dilation
                    temp_map = binary_dilation(mask_data > 128, iterations=step_d)
                    _, n_groups = label(temp_map)
                    # Check condition: Do we have 4 or fewer clusters?
                    if n_groups <= 4:
                        final_dilation_iter = step_d
                        print(f"  [Auto-Group] Converged at dilation: {step_d}px (Groups: {n_groups})")
                        found_optimal = True
                        break
                    # Update "best so far" (in case we never hit <=4, we use the max tested)
                    final_dilation_iter = step_d

                if not found_optimal:
                    print(f"  [Auto-Group] Max dilation limit reached ({final_dilation_iter}px). Proceeding.")
            else:
                # Standard fixed dilation for normal mosaic (usually denser)
                final_dilation_iter = 30
                
            group_map = binary_dilation(mask_data > 128, iterations=final_dilation_iter)
            labeled_groups, num_groups = label(group_map)
            coverage_mask = np.zeros((h, w), dtype=bool)

            # 1. Count pixels for every group ID efficiently
            # group_ids will contain [0, 1, 2...] and counts will be their pixel sizes
            group_ids, group_sizes = np.unique(labeled_groups, return_counts=True)
            # 2. Store (id, size) tuples, ignoring ID 0 (background)
            valid_groups = []
            for g_id, g_size in zip(group_ids, group_sizes):
                if g_id == 0: continue # Skip background
                valid_groups.append((g_id, g_size))
            
            # 3. Sort by size (largest first) and slice the top 5
            valid_groups.sort(key=lambda x: x[1], reverse=True)
            target_groups = valid_groups[:5]

            print(f"  > Found {num_groups} mask clusters. Processing top {len(target_groups)} largest.")
            for group_id, size in target_groups:
                # Mask for the current CLUSTER
                current_group_indices = (labeled_groups == group_id)
                # Intersection of Group Box and Actual Mask
                active_pixels_in_group = (mask_data > 128) & current_group_indices
                
                if not np.any(active_pixels_in_group): continue
                # We skip the check below because each group_id is unique by definition
                # if np.all(coverage_mask[active_pixels_in_group]): continue

                # Center of Mass & Bounding Box of the CLUSTER
                y_center, x_center = center_of_mass(current_group_indices)
                
                # Determine crop size based on target resolution
                current_inpaint_size = self.target_resolution 
                ax1, ay1 = max(0, int(x_center) - current_inpaint_size // 2), max(0, int(y_center) - current_inpaint_size // 2)
                ax2, ay2 = min(w, ax1 + current_inpaint_size), min(h, ay1 + current_inpaint_size)
                
                # Boundary Checks
                if ax2 - ax1 < current_inpaint_size:
                    if ax1 == 0: ax2 = min(w, current_inpaint_size)
                    if ax2 == w: ax1 = max(0, w - current_inpaint_size)
                if ay2 - ay1 < current_inpaint_size:
                    if ay1 == 0: ay2 = min(h, current_inpaint_size)
                    if ay2 == h: ay1 = max(0, h - current_inpaint_size)

                tmp_canvas = Image.fromarray((output_canvas).astype(np.uint8))
                adaptive_tile = tmp_canvas.crop((ax1, ay1, ax2, ay2))
                group_mask_pil = Image.fromarray((active_pixels_in_group * 255).astype(np.uint8), mode='L')
                adaptive_mask_img = group_mask_pil.crop((ax1, ay1, ax2, ay2))

                # Standardize inputs (SD requirement)
                st_tw, st_th = (adaptive_tile.width // 64) * 64, (adaptive_tile.height // 64) * 64
                if st_tw == 0 or st_th == 0: continue
                scale_factor = 1.0
                if self.target_resolution > 512:
                     if max(st_tw, st_th) < 768:
                         scale_factor = min(2.0, self.target_resolution / max(st_tw, st_th))
                
                final_tw, final_th = int(st_tw * scale_factor), int(st_th * scale_factor)
                final_tw, final_th = (final_tw // 8) * 8, (final_th // 8) * 8 # Ensure div by 8

                tile_st = adaptive_tile.resize((final_tw, final_th), Image.LANCZOS)
                mask_st = adaptive_mask_img.resize((final_tw, final_th), Image.NEAREST)
                
                # INPAINT
                inpainted_st = self.pipeline(
                    prompt=prompts["prompt"],
                    negative_prompt=prompts["negative_prompt"],
                    image=tile_st, mask_image=mask_st,
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=7.5,
                    strength=adaptive_strength,
                ).images[0]
                inpainted_final = inpainted_st.resize(adaptive_tile.size, Image.LANCZOS)
                inpainted_final = ImageOps.grayscale(inpainted_final).convert("RGB")
                mask_array = np.array(adaptive_mask_img) > 128
                dilated_mask_array = binary_dilation(mask_array, iterations=final_dilation_iter)
                expanded_mask = Image.fromarray((dilated_mask_array * 255).astype(np.uint8), mode='L')
                # 4. Smooth the edges to prevent harsh seams where the inpainted area meets the original image
                # (You had this commented out, but it's highly recommended for a natural blend)
                blend_mask = expanded_mask.filter(ImageFilter.GaussianBlur(radius=5))
                result_tile = Image.composite(inpainted_final, adaptive_tile, blend_mask)
                self.save_debug_images(idx, group_id, tile_st, mask_st, result_tile)
                output_canvas[ay1:ay2, ax1:ax2] = np.array(result_tile).astype(np.float32)


            final_img = ImageOps.grayscale(Image.fromarray(np.clip(output_canvas, 0, 255).astype(np.uint8))).convert("RGB")
            final_img.save(os.path.join(self.args.output, f"{idx}_decensored.png"))


# --- NEW IMPROVED INPAINTER ---
class DecensorInpainter2(DecensorInpainter):
    """
    Advanced Inpainter using 'NoobAI XL 1.1'.
    A modern (2025 era) model fine-tuned specifically on Danbooru tags.
    It excels at 'lineart' and 'monochrome' styles where Pony models might force 3D/color.
    """
    def __init__(self, args):
        super().__init__(args)
        # NoobAI is SDXL based, so it prefers 1024px
        self.target_resolution = INPAINT_SIZE 

    def load_pipeline(self):
        if self.pipeline is None:
            print("Initializing Advanced Pipeline (NoobAI XL 1.1)...")
            # We use the Base SDXL pipeline -> Inpainting Adapter strategy
            # This is robust for models that are distributed as single checkpoints
            
            repo_id = "Laxhar/noobai-XL-1.1"
            ckpt_name = "noobai-XL-1.1.safetensors"
            
            # Local cache check
            local_models_dir = os.path.join(os.getcwd(), "models")
            if not os.path.exists(local_models_dir):
                os.makedirs(local_models_dir, exist_ok=True)
                
            local_ckpt = os.path.join(local_models_dir, ckpt_name)
            
            ckpt_path = ""
            if os.path.exists(local_ckpt):
                print(f"Loading local checkpoint: {local_ckpt}")
                ckpt_path = local_ckpt
            else:
                print(f"Downloading NoobAI XL 1.1 from HF: {repo_id}")
                try:
                    # Attempt to download the single file if possible
                    # Note: You might need to adjust filename if the repo structure changes
                    # This model is often just 'model.safetensors' or similar in diffusers repo, 
                    # but usually shared as a single file in 'Laxhar/noobai-XL-1.1'
                    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
                except Exception:
                    print("Could not download specific filename, trying default diffusers load...")
                    # Fallback to standard diffusers load if it is a repo
                    self.pipeline = AutoPipelineForInpainting.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    ).to("cuda")
                    self.pipeline.enable_vae_slicing()
                    return

            # If we got a checkpoint path, load via SingleFile
            print("Loading SDXL Base Pipeline from single file...")
            temp_pipe = StableDiffusionXLPipeline.from_single_file(
                ckpt_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            print("Converting to AutoPipelineForInpainting...")
            self.pipeline = AutoPipelineForInpainting.from_pipe(temp_pipe).to("cuda")
            self.pipeline.enable_vae_slicing()
            
            del temp_pipe
            gc.collect()
            torch.cuda.empty_cache()

    def get_prompts(self):
        # Prompts optimized for NoobAI XL & Monochrome Manga
        # NoobAI responds very well to "masterpiece, best quality" + danbooru tags
        return {
            "prompt": "masterpiece, best quality, monochrome, greyscale, lineart, manga, comic, sketchy, detailed anatomy, uncensored, penis, vagina, genitals, (white background:1.2), (simple background:1.1)",
            "negative_prompt": "color, colored, 3d, realistic, photorealistic, volumetric lighting, painting, acrylic, source_pony, source_furry, mosaic, censor, bar, blurry, text, watermark, bad anatomy, bad hands, extra digits"
        }