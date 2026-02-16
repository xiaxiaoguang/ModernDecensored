import os
import random
import cv2
import numpy as np
import torch
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from safetensors.torch import save_file 
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline
)

class RobustInpaintDataset(Dataset):
    def __init__(self, root_dir, resolution=1024, mode="hybrid", augment_mask=True):
        self.root_dir = root_dir
        self.augment_mask = augment_mask
        self.res = resolution
        
        # Determine subtasks
        if mode == "bar":
            self.sub_tasks = ['inpainter_bar']
        elif mode == "mosaic":
            self.sub_tasks = ['inpainter_mosaic']
        else:
            self.sub_tasks = ['inpainter_bar', 'inpainter_mosaic']
            
        print(f"[Dataset] Initialized with Mode: {mode.upper()} | Robust Masking: {augment_mask}")
        
        self.image_entries = []
        for sub in self.sub_tasks:
            sub_gt_path = os.path.join(root_dir, sub, "ground_truth")
            if not os.path.exists(sub_gt_path): continue
            files = [f for f in os.listdir(sub_gt_path) if f.endswith('.png')]
            for f in files:
                self.image_entries.append((sub, f))

        self.img_tf = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Mask transforms
        self.resize = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST)
        self.crop = transforms.CenterCrop(resolution)

    def _augment_mask_logic(self, mask_img, censored_img):
        """ Applies custom manga-specific augmentations to the mask """
        # Convert PIL images to numpy arrays for complex manipulation
        mask_np = np.array(mask_img)
        censored_np = np.array(censored_img)
        
        # Ensure mask is strictly binary (0 or 255)
        mask_binary = (mask_np > 127).astype(np.uint8) * 255
        
        # ==========================================
        # LOGIC 1: Simulate Imperfect Detection 
        # (Drop, Shrink, or Expand individual blocks)
        # ==========================================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        
        new_mask = np.zeros_like(mask_binary)
        
        # Iterate through each connected mask component (skip label 0, which is background)
        for i in range(1, num_labels):
            comp_mask = (labels == i).astype(np.uint8) * 255
            
            # Decide fate of this specific mask block
            # Probabilities: 15% Drop, 35% Shrink, 35% Expand, 15% Keep intact
            action_roll = random.random()
            
            if action_roll < 0.15:
                # 1. Undetected mask (Drop entirely)
                continue 
            
            elif action_roll < 0.50:
                # 2. Detected but inaccurate - Shrink (erosion)
                k = random.randrange(3, 15, 2)
                kernel = np.ones((k, k), np.uint8)
                comp_mask = cv2.erode(comp_mask, kernel, iterations=1)
                
            elif action_roll < 0.85:
                # 3. Detected but inaccurate - Expand (dilation)
                k = random.randrange(3, 15, 2)
                kernel = np.ones((k, k), np.uint8)
                comp_mask = cv2.dilate(comp_mask, kernel, iterations=1)
                
            # If roll >= 0.85, keep intact. Add the component to the new mask canvas.
            new_mask = cv2.bitwise_or(new_mask, comp_mask)

        # ==========================================
        # LOGIC 2: Pre-rendering strange overlays 
        # (White bubbles/liquid crossing the bar)
        # ==========================================
        if random.random() < 0.7:  # 70% chance to apply this logic
            # Detect pure or near-white pixels in the censored image
            # Usually manga white liquids/bubbles are > 240 in RGB
            white_thresh = 240
            is_white = (censored_np[:, :, 0] > white_thresh) & \
                       (censored_np[:, :, 1] > white_thresh) & \
                       (censored_np[:, :, 2] > white_thresh)
            
            white_mask = is_white.astype(np.uint8) * 255
            # Optional: Dilate the white areas slightly to account for anti-aliasing/soft edges 
            # around the liquid so the model doesn't try to inpaint the borders of the liquid
            kernel = np.ones((3, 3), np.uint8)
            white_mask = cv2.dilate(white_mask, kernel, iterations=1)
            # Unmask those white pixel regions (set mask to 0 where liquid exists)
            new_mask[white_mask == 255] = 0

        return Image.fromarray(new_mask)

    def __getitem__(self, idx):
        sub_folder, fn = self.image_entries[idx]
        
        gt_path = os.path.join(self.root_dir, sub_folder, "ground_truth", fn)
        censored_path = os.path.join(self.root_dir, sub_folder, "censored", fn)
        mask_path = os.path.join(self.root_dir, sub_folder, "mask", fn)

        gt = Image.open(gt_path).convert("RGB")
        censored = Image.open(censored_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment_mask:
            # Pass both mask and censored image to our new augmentation method
            mask = self._augment_mask_logic(mask, censored)
        # mask.save('1.png')

        return {
            "pixel_values": self.img_tf(gt),
            "mask_values": transforms.ToTensor()(self.crop(self.resize(mask))),
            "masked_image_values": self.img_tf(censored),
            "filename": fn
        }

    def __len__(self):
        return len(self.image_entries)

def save_compatible_lora(unet, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state_dict = unet.state_dict()
    peft_dict = {k: v for k, v in state_dict.items() if "lora" in k}
    diffusers_dict = {}
    for k, v in peft_dict.items():
        new_k = k.replace("base_model.model.", "")
        if "lora_A" in new_k: new_k = new_k.replace("lora_A", "lora.down")
        elif "lora_B" in new_k: new_k = new_k.replace("lora_B", "lora.up")
        new_k = re.sub(r"\.default(_\d+)?", "", new_k)
        if not new_k.startswith("unet."): new_k = f"unet.{new_k}"
        diffusers_dict[new_k] = v
    save_file(diffusers_dict, os.path.join(output_dir, "pytorch_lora_weights.safetensors"))
    
def visualize_results(model_path, lora_folder, dataset, output_path, num_samples=10):
    print(f"\n[Visual Test] Generating Validation Samples...")
    torch.cuda.empty_cache()
    try:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        pipe.load_lora_weights(lora_folder, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora()
    except Exception as e:
        print(f"[Visual Test] Error loading pipeline: {e}")
        return

    os.makedirs(output_path, exist_ok=True)
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        sub_folder, filename = dataset.image_entries[idx]
        
        if "mosaic" in sub_folder:
            denoising_strength = 0.66
        else:
            denoising_strength = 0.99 

        full_censored_path = os.path.join(dataset.root_dir, sub_folder, "censored", filename)
        full_mask_path = os.path.join(dataset.root_dir, sub_folder, "mask", filename)
        full_gt_path = os.path.join(dataset.root_dir, sub_folder, "ground_truth", filename)

        censored = Image.open(full_censored_path).convert("RGB").resize((1024, 1024))
        mask = Image.open(full_mask_path).convert("L").resize((1024, 1024))
        gt = Image.open(full_gt_path).convert("RGB").resize((1024, 1024))
        
        result = pipe(
            prompt="reconstruct, genital detail, high quality, uncensored, lineart, manga style",
            negative_prompt="mosaic, black bars, censor, error, blurry, low quality",
            image=censored,
            mask_image=mask,
            num_inference_steps=35, 
            guidance_scale=7.5,
            strength=denoising_strength 
        ).images[0]
        
        w, h = censored.size
        grid = Image.new("RGB", (w * 4, h))
        grid.paste(censored, (0, 0))
        grid.paste(mask.convert("RGB"), (w, 0))
        grid.paste(gt, (w * 2, 0))
        grid.paste(result, (w * 3, 0))
        grid.save(os.path.join(output_path, f"val_{filename}_s{int(denoising_strength*100)}.png"))
        
    del pipe
    torch.cuda.empty_cache()
