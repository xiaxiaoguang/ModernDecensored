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

from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, AutoPipelineForInpainting, StableDiffusionXLPipeline
from transformers import Sam2Processor, Sam2Model
from .tools import * 
from .inpainter import DecensorInpainter

# --- NEW IMPROVED INPAINTER ---
class DecensorInpainterX(DecensorInpainter):
    """
    Advanced Inpainter using Pony Diffusion V6 XL with Morphological Grouping.
    Uses StableDiffusionXLPipeline.from_single_file loading strategy converted to Inpainting.
    """
    def load_pipeline(self):
        if self.pipeline is None:
            print("Initializing Advanced Inpainting Pipeline (Pony Diffusion V6 XL)...")
            
            repo_id = "LyliaEngine/Pony_Diffusion_V6_XL"
            ckpt_name = "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
            vae_name = "sdxl_vae.safetensors"
            
            # 1. Path Management: Check local 'models' folder first, else use HF Cache
            local_models_dir = os.path.join(os.getcwd(), "models")
            local_ckpt = os.path.join(local_models_dir, ckpt_name)
            local_vae = os.path.join(local_models_dir, vae_name)

            ckpt_path = ""
            vae_path = ""

            # Check Checkpoint
            if os.path.exists(local_ckpt):
                print(f"Loading local checkpoint: {local_ckpt}")
                ckpt_path = local_ckpt
            else:
                print(f"Local checkpoint not found, downloading from HF: {repo_id}")
                ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)

            # Check VAE
            if os.path.exists(local_vae):
                print(f"Loading local VAE: {local_vae}")
                vae_path = local_vae
            else:
                print(f"Downloading VAE from HF...")
                vae_path = hf_hub_download(repo_id=repo_id, filename=vae_name)

            try:
                # 2. Load as Standard SDXL Pipeline (Text-to-Image)
                # This bypasses the 'AutoPipeline' limitation with single files
                print("Loading SDXL Base Pipeline from single file...")
                temp_pipe = StableDiffusionXLPipeline.from_single_file(
                    ckpt_path,
                    vae_path=vae_path,
                    torch_dtype=torch.float16
                )

                # 3. Convert to Inpainting Pipeline
                # This wrapper will adapt the base model to accept 'mask_image' inputs
                print("Converting to AutoPipelineForInpainting...")
                self.pipeline = AutoPipelineForInpainting.from_pipe(temp_pipe).to("cuda")
                
                # Cleanup temp pipe logic just in case, though from_pipe handles it
                del temp_pipe
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nCRITICAL ERROR loading model: {e}")
                raise e
            
            # Enable memory optimizations
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_model_cpu_offload() 

    def inpaint_process(self):
        self.load_pipeline()
        if not os.path.exists(self.args.output): os.makedirs(self.args.output)
        
        # Pony handles strength differently. 1.0 is good for bars, 0.75-0.85 for mosaics.
        adaptive_strength = 0.80 if self.args.mosaic == 1 else 1.0

        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        # Natural sort
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

        for idx, orig_file in enumerate(input_files):
            print(f"Processing [Pony]: {orig_file}")
            orig_path = os.path.join(self.args.input, orig_file)
            
            # Load and Resize
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            
            # Optional: Preprocess if black levels are crushed
            if self.args.black_level > 0:
                img_for_det = self.preprocess_for_detection(original_img.copy())
            else:
                img_for_det = original_img

            # --- MASK LOADING LOGIC ---
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
                        except Exception as e:
                            continue
            
            if not full_mask.getbbox():
                print(f"No mask found for {orig_file}, saving original.")
                original_img.save(os.path.join(self.args.output, f"{orig_file}"))
                continue

            # --- INPAINTING LOGIC WITH MORPHOLOGICAL GROUPING ---
            output_canvas = np.array(original_img).astype(np.float32)
            mask_data = np.array(full_mask)
            
            # 1. GROUPING STRATEGY: Morphological Closing
            # We dilate the mask heavily to bridge gaps between bars.
            grouping_dilation_iter = 15 
            
            group_map = binary_dilation(mask_data > 128, iterations=grouping_dilation_iter)
            labeled_groups, num_groups = label(group_map)
            coverage_mask = np.zeros((h, w), dtype=bool)

            print(f"Found {num_groups} distinct censored regions (clusters).")

            for group_id in range(1, num_groups + 1):
                # Mask for the current CLUSTER
                current_group_indices = (labeled_groups == group_id)
                active_pixels_in_group = (mask_data > 128) & current_group_indices
                
                if not np.any(active_pixels_in_group): continue
                if np.all(coverage_mask[active_pixels_in_group]): continue

                # Center of Mass & Bounding Box
                y_center, x_center = center_of_mass(current_group_indices)
                slices = find_objects(labeled_groups == group_id)
                if not slices: continue
                y_slice, x_slice = slices[0]
                
                # PONY OPTIMIZATION: Always use 1024x1024 context if possible
                current_inpaint_size = INPAINT_SIZE
                
                ax1, ay1 = max(0, int(x_center) - current_inpaint_size // 2), max(0, int(y_center) - current_inpaint_size // 2)
                ax2, ay2 = min(w, ax1 + current_inpaint_size), min(h, ay1 + current_inpaint_size)
                
                # Correct crop handling for edges
                if ax2 - ax1 < current_inpaint_size:
                    if ax1 == 0: ax2 = min(w, current_inpaint_size)
                    if ax2 == w: ax1 = max(0, w - current_inpaint_size)
                if ay2 - ay1 < current_inpaint_size:
                    if ay1 == 0: ay2 = min(h, current_inpaint_size)
                    if ay2 == h: ay1 = max(0, h - current_inpaint_size)

                adaptive_tile = original_img.crop((ax1, ay1, ax2, ay2))
                adaptive_mask_img = full_mask.crop((ax1, ay1, ax2, ay2))

                # Standardize inputs to multiple of 8
                st_tw, st_th = (adaptive_tile.width // 64) * 64, (adaptive_tile.height // 64) * 64
                
                # Upscale small tiles
                if st_tw < 512 or st_th < 512:
                    scale_factor = max(1024/max(1, st_tw), 1024/max(1, st_th))
                    scale_factor = min(scale_factor, 4.0) 
                    st_tw, st_th = int(st_tw * scale_factor), int(st_th * scale_factor)
                    st_tw, st_th = (st_tw // 64) * 64, (st_th // 64) * 64

                tile_st = adaptive_tile.resize((st_tw, st_th), Image.LANCZOS)
                mask_st = adaptive_mask_img.resize((st_tw, st_th), Image.NEAREST)
                # mask_st = mask_st.filter(ImageFilter.MaxFilter(11)).filter(ImageFilter.GaussianBlur(5))

                # --- PONY PROMPTING ---
                pony_prompt = (
                    "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, "
                    "monochrome, greyscale, manga style, lineart, "
                    "uncensored, detailed anatomy, genitals, "
                    "simple background, white background"
                )
                
                pony_negative = (
                    "score_1, score_2, score_3, "
                    "source_pony, source_furry, 3d, realistic, photo, "
                    "mosaic, bar censor, censorship, blurry, text, artist name, color"
                )

                inpainted_st = self.pipeline(
                    prompt=pony_prompt,
                    negative_prompt=pony_negative,
                    image=tile_st, 
                    mask_image=mask_st,
                    num_inference_steps=30, 
                    guidance_scale=7.0,
                    strength=adaptive_strength 
                ).images[0]
                
                # Resize back and Blend
                inpainted_final = inpainted_st.resize(adaptive_tile.size, Image.LANCZOS)
                inpainted_final = ImageOps.grayscale(inpainted_final).convert("RGB")
                self.save_debug_images(idx, group_id, tile_st, mask_st, inpainted_st)
                blend_mask = adaptive_mask_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(4))
                result_tile = Image.composite(inpainted_final, adaptive_tile, blend_mask)
                
                output_canvas[ay1:ay2, ax1:ax2] = np.array(result_tile).astype(np.float32)
                coverage_mask[ay1:ay2, ax1:ax2] = True

            final_res = Image.fromarray(np.clip(output_canvas, 0, 255).astype(np.uint8))
            final_res = ImageOps.grayscale(final_res).convert("RGB")
            final_res.save(os.path.join(self.args.output, f"{orig_file}_decensored"))

        print("Pony Inpainting Process Complete.")




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
        self.target_resolution = 1024 

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