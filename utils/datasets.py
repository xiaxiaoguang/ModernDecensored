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
    
    def __len__(self):
        return len(self.image_entries)
    
    def _augment_mask_logic(self, mask_img, censored_img):
        """ 
        Applies manga-specific augmentations to the mask and dynamically 
        draws adversarial occlusions onto the censored image.
        """
        mask_np = np.array(mask_img)
        censored_np = np.array(censored_img)
        
        mask_binary = (mask_np > 127).astype(np.uint8) * 255
        
        # ==========================================
        # LOGIC 1: Simulate Imperfect Detection 
        # ==========================================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        new_mask = np.zeros_like(mask_binary)
        
        for i in range(1, num_labels):
            comp_mask = (labels == i).astype(np.uint8) * 255
            action_roll = random.random()
            
            if action_roll < 0.15:
                continue # Drop
            elif action_roll < 0.50:
                k = random.randrange(3, 15, 2)
                kernel = np.ones((k, k), np.uint8)
                comp_mask = cv2.erode(comp_mask, kernel, iterations=1) # Shrink
            elif action_roll < 0.85:
                k = random.randrange(3, 15, 2)
                kernel = np.ones((k, k), np.uint8)
                comp_mask = cv2.dilate(comp_mask, kernel, iterations=1) # Expand
                
            new_mask = cv2.bitwise_or(new_mask, comp_mask)

        # ==========================================
        # LOGIC 2: Dynamic Manga Occlusion Generation
        # (Draw effects ON the image, REMOVE from mask)
        # ==========================================
        if random.random() < 0.7:  
            # Find the active mask areas so we know where to drop our fake effects
            active_y, active_x = np.where(new_mask > 0)
            
            if len(active_x) > 0:
                # Pick a random point currently inside the black bar mask
                idx = random.randint(0, len(active_x) - 1)
                cx, cy = active_x[idx], active_y[idx]
                
                effect_mask = np.zeros_like(new_mask)
                gray_val = random.randint(180, 255)
                color = (gray_val, gray_val, gray_val)
                
                # Draw an organic liquid splatter
                for _ in range(random.randint(2, 6)):
                    drop_x = cx + random.randint(-30, 30)
                    drop_y = cy + random.randint(-30, 30)
                    radius = random.randint(10, 30)
                    
                    pts = []
                    for angle in range(0, 360, 45):
                        rad = np.deg2rad(angle)
                        dist = radius + random.randint(-radius//2, radius//2)
                        pts.append([int(drop_x + dist * np.cos(rad)), int(drop_y + dist * np.sin(rad))])
                    
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    # Draw onto our blank tracking mask
                    cv2.fillPoly(effect_mask, [pts], 255)
                    # Draw the actual gray blob onto the censored image
                    cv2.fillPoly(censored_np, [pts], color)
                    cv2.polylines(censored_np, [pts], True, (0,0,0), 2)
                
                # Subtract the effect shape from the training mask.
                # This teaches the model: "Do not touch the liquid, only paint behind it!"
                new_mask[effect_mask == 255] = 0

        return Image.fromarray(new_mask), Image.fromarray(censored_np)

    def __getitem__(self, idx):
        sub_folder, fn = self.image_entries[idx]
        
        gt_path = os.path.join(self.root_dir, sub_folder, "ground_truth", fn)
        censored_path = os.path.join(self.root_dir, sub_folder, "censored", fn)
        mask_path = os.path.join(self.root_dir, sub_folder, "mask", fn)

        gt = Image.open(gt_path).convert("RGB")
        censored = Image.open(censored_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment_mask:
            # We now catch BOTH the augmented mask AND the freshly painted censored image
            mask, censored = self._augment_mask_logic(mask, censored)

        return {
            "pixel_values": self.img_tf(gt),
            "mask_values": transforms.ToTensor()(self.crop(self.resize(mask))),
            "masked_image_values": self.img_tf(censored),
            "filename": fn
        }

class SAM2Dataset(Dataset):
    def __init__(self, root_dir, processor, mode="bar", augment=True):
        self.root_dir = root_dir
        self.processor = processor
        self.augment = augment
        self.entries = []
        
        search_path = os.path.join(root_dir, f"inpainter_{mode}", "censored")
        mask_root = os.path.join(root_dir, f"inpainter_{mode}", "mask")
        
        if os.path.exists(search_path):
            files = [f for f in os.listdir(search_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                mask_path = os.path.join(mask_root, f)
                if os.path.exists(mask_path):
                    self.entries.append({
                        "image_path": os.path.join(search_path, f),
                        "mask_path": mask_path
                    })
        
        print(f"[*] Found {len(self.entries)} total pairs. | Augmentation: {augment}")

    def _apply_manga_occlusions(self, img_np, mask_np):
        """
        Dynamically draws adversarial effects (bubbles, liquid) on the image
        to teach SAM 2 to ignore overlapping manga elements.
        """
        aug_img = img_np.copy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Find individual censor bars using connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        for i in range(1, num_labels):
            # 60% chance to occlude this specific bar
            if random.random() < 0.6: 
                x, y, w, h, area = stats[i]
                
                # Prevent crashing on tiny noise specs
                if w < 5 or h < 5: continue 
                
                effect = random.choice(['bubble', 'droplets', 'screentone_slash'])
                gray_val = random.randint(180, 255)
                color = (gray_val, gray_val, gray_val)

                if effect == 'bubble':
                    # Bubble taking a bite out of a corner
                    corner_x = random.choice([x, x + w])
                    corner_y = random.choice([y, y + h])
                    r_w = random.randint(max(2, int(w * 0.2)), max(3, int(w * 0.6)))
                    r_h = random.randint(max(2, int(h * 0.2)), max(3, int(h * 0.6)))
                    cv2.ellipse(aug_img, (corner_x, corner_y), (r_w, r_h), 0, 0, 360, color, -1)
                    cv2.ellipse(aug_img, (corner_x, corner_y), (r_w, r_h), 0, 0, 360, (0, 0, 0), 2)
                    
                elif effect == 'droplets':
                    # Organic liquid splatter
                    for _ in range(random.randint(3, 8)):
                        drop_x = random.randint(x, x + w)
                        drop_y = random.randint(y, y + h)
                        radius = random.randint(4, max(5, w // 6 + 1))
                        pts = []
                        for angle in range(0, 360, 45):
                            rad = np.deg2rad(angle)
                            dist = radius + random.randint(-radius//2, radius//2)
                            pts.append([int(drop_x + dist * np.cos(rad)), int(drop_y + dist * np.sin(rad))])
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(aug_img, [pts], color)
                        cv2.polylines(aug_img, [pts], True, (0,0,0), 1)
                        
                elif effect == 'screentone_slash':
                    # Layer slash
                    start_y = random.randint(y, y + h)
                    end_y = random.randint(y, y + h)
                    thickness = random.randint(3, max(4, h // 5 + 1))
                    cv2.line(aug_img, (max(0, x - 15), start_y), (min(img_np.shape[1], x + w + 15), end_y), color, thickness)

        return aug_img

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        image = Image.open(entry["image_path"]).convert("RGB")
        gt_mask = Image.open(entry["mask_path"]).convert("L")
        
        image_np = np.array(image)
        mask_np = np.array(gt_mask) > 128
        
        # Apply data augmentation to the IMAGE only. 
        # The ground truth mask remains PERFECT to teach SAM object permanence.
        if self.augment:
            image_np = self._apply_manga_occlusions(image_np, mask_np)
            image = Image.fromarray(image_np)
        
        # Simulate Prompt (Point selection)
        y_indices, x_indices = np.where(mask_np)
        if len(y_indices) > 0:
            rand_idx = random.randint(0, len(y_indices) - 1)
            prompt_point = [int(x_indices[rand_idx]), int(y_indices[rand_idx])]
            prompt_label = 1 
        else:
            prompt_point = [512, 512]
            prompt_label = 0

        inputs = self.processor(
            images=image, 
            input_points=[[[prompt_point]]], 
            input_labels=[[[prompt_label]]], 
            return_tensors="pt"
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        gt_mask_resized = gt_mask.resize((256, 256), Image.NEAREST)
        inputs["ground_truth_mask"] = torch.tensor(np.array(gt_mask_resized) > 128).float()

        return inputs
    
def synthesize_manga_occlusions(img_path, txt_path, out_img_path):
    """
    Reads YOLO labels and draws realistic adversarial manga occlusions 
    (gray bubbles, organic liquid blobs, screentones) over the bars.
    """
    img = cv2.imread(img_path)
    if img is None: return False
    h, w = img.shape[:2]
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        class_id = int(parts[0])
        
        # Only apply occlusions to Class 1 (Black Bars)
        if class_id == 1: 
            cx, cy, bw, bh = map(float, parts[1:5])
            x_center, y_center = int(cx * w), int(cy * h)
            box_w, box_h = int(bw * w), int(bh * h)
            
            x1 = max(0, x_center - box_w // 2)
            y1 = max(0, y_center - box_h // 2)
            x2 = min(w, x_center + box_w // 2)
            y2 = min(h, y_center + box_h // 2)
            
            # 80% chance to apply an adversarial effect
            if random.random() < 0.8: 
                effect = random.choice(['bubble', 'droplets', 'screentone_slash'])
                
                # Random grayscale intensity (mimicking different manga shading)
                gray_val = random.randint(200, 255)
                fill_color = (gray_val, gray_val, gray_val)
                border_color = (max(0, gray_val - 100), max(0, gray_val - 100), max(0, gray_val - 100))
                
                if effect == 'bubble':
                    # Simulate a speech bubble overlapping a corner
                    corner_x = random.choice([x1, x2])
                    corner_y = random.choice([y1, y2])
                    r_w = random.randint(int(box_w * 0.2), int(box_w * 0.6))
                    r_h = random.randint(int(box_h * 0.2), int(box_h * 0.6))
                    
                    cv2.ellipse(img, (corner_x, corner_y), (r_w, r_h), 0, 0, 360, fill_color, -1)
                    cv2.ellipse(img, (corner_x, corner_y), (r_w, r_h), 0, 0, 360, border_color, 2)
                    
                elif effect == 'droplets':
                    # Simulate organic, irregular liquid splatters
                    for _ in range(random.randint(3, 8)):
                        drop_x = random.randint(x1, x2)
                        drop_y = random.randint(y1, y2)
                        radius = random.randint(4, max(8, box_w // 6))
                        
                        # Generate an irregular polygon instead of a perfect circle
                        pts = []
                        for angle in range(0, 360, 45):
                            rad = np.deg2rad(angle)
                            dist = radius + random.randint(-radius//2, radius//2)
                            pts.append([int(drop_x + dist * np.cos(rad)), int(drop_y + dist * np.sin(rad))])
                        
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(img, [pts], fill_color)
                        cv2.polylines(img, [pts], True, border_color, 1)
                        
                elif effect == 'screentone_slash':
                    # Simulate a jagged cut filled with a fake screentone pattern
                    start_y = random.randint(y1, y2)
                    end_y = random.randint(y1, y2)
                    thickness = random.randint(4, max(8, box_h // 5))
                    
                    # Create a temporary mask for the slash
                    temp_mask = np.zeros_like(img)
                    cv2.line(temp_mask, (max(0, x1 - 15), start_y), (min(w, x2 + 15), end_y), (255, 255, 255), thickness)
                    
                    # Apply a dotted texture where the mask is active
                    grid_size = 4
                    for y in range(0, h, grid_size):
                        for x in range(0, w, grid_size):
                            if temp_mask[y, x, 0] > 0:
                                cv2.circle(img, (x, y), 1, (100, 100, 100), -1)

    cv2.imwrite(out_img_path, img)
    return True

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


def visualize_results(model_path, lora_folder, dataset, output_path, num_samples=10, epoch=0, val_loss=0):
    
    import csv
    # --- 1. CSV LOGGING ---
    csv_path = os.path.join(output_path, "val_loss.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header if it's the first time
        if not file_exists:
            writer.writerow(["Epoch", "Validation_Loss"])
        writer.writerow([epoch + 1, val_loss])

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
