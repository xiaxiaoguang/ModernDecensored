import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import AdamW
from tqdm import tqdm

# --- TRANSFORMERS IMPORTS ---
from transformers import Sam2Model, Sam2Processor

# --- CONFIGURATION ---
MODEL_ID = "./checkpoints" 
DATASET_ROOT = "./dataset_refined"
OUTPUT_DIR = "./sam2_transformers_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Config
BATCH_SIZE = 1 
GRADIENT_ACCUMULATION_STEPS = 8 
LR = 2e-6 
EPOCHS = 60

class SAM2Dataset(Dataset):
    def __init__(self, root_dir, processor, mode="bar"):
        self.root_dir = root_dir
        self.processor = processor
        self.entries = []
        
        search_path = os.path.join(root_dir, f"inpainter_{mode}", "censored")
        mask_root = os.path.join(root_dir, f"inpainter_{mode}", "mask")
        
        if os.path.exists(search_path):
            files = [f for f in os.listdir(search_path) if f.endswith('.png') or f.endswith('.jpg')]
            for f in files:
                mask_path = os.path.join(mask_root, f)
                if os.path.exists(mask_path):
                    self.entries.append({
                        "image_path": os.path.join(search_path, f),
                        "mask_path": mask_path
                    })
        
        print(f"[*] Found {len(self.entries)} total pairs.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        image = Image.open(entry["image_path"]).convert("RGB")
        gt_mask = Image.open(entry["mask_path"]).convert("L")
        
        image_np = np.array(image)
        mask_np = np.array(gt_mask) > 128
        
        # Simulate Prompt
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

def compute_iou(pred_mask, gt_mask):
    """Calculates Intersection over Union for binary masks"""
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    if union == 0:
        return 1.0
    return intersection / union

def evaluate_model(model, dataloader, desc="Evaluating"):
    """
    Runs evaluation on the test set and calculates IoU.
    Does NOT update gradients.
    """
    model.eval()
    total_iou = 0
    total_loss = 0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_points = batch["input_points"].to(DEVICE)
            input_labels = batch["input_labels"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE).unsqueeze(1) 
            
            outputs = model(
                pixel_values=pixel_values, 
                input_points=input_points,
                input_labels=input_labels,
                multimask_output=False 
            )
            
            pred_masks = outputs.pred_masks.squeeze(1)
            loss = loss_fn(pred_masks, gt_masks)
            
            # Calculate metrics
            iou = compute_iou(pred_masks, gt_masks)
            
            total_loss += loss.item()
            total_iou += iou.item()
            count += 1
            
    avg_loss = total_loss / count
    avg_iou = total_iou / count
    
    model.train() # Switch back to training
    return avg_loss, avg_iou

def visualize_validation(model, dataset, epoch, output_dir, num_samples=3):
    """
    Generates visuals. Handles both standard Dataset and random_split Subsets.
    """
    print(f"\n[Visualizer] Generating {num_samples} TEST images...")
    model.eval()
    
    vis_dir = os.path.join(output_dir, "visuals_epoch_" + str(epoch+1))
    os.makedirs(vis_dir, exist_ok=True)
    
    # Handle Subset logic from random_split
    if isinstance(dataset, Subset):
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        # We need to map the Subset index back to the Original Dataset index to get file paths
        original_indices = [dataset.indices[i] for i in indices]
        parent_dataset = dataset.dataset
    else:
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        original_indices = indices
        parent_dataset = dataset

    for i, idx in enumerate(indices):
        # 1. Get File Info (Need to access parent dataset for paths)
        real_idx = original_indices[i]
        entry = parent_dataset.entries[real_idx]
        
        image_raw = Image.open(entry["image_path"]).convert("RGB")
        gt_mask_raw = Image.open(entry["mask_path"]).convert("L")
        
        # 2. Get Tensors via __getitem__ (Using the wrapper dataset)
        processed_inputs = dataset[idx] 
        
        pixel_values = processed_inputs["pixel_values"].unsqueeze(0).to(DEVICE)
        input_points = processed_inputs["input_points"].unsqueeze(0).to(DEVICE)
        input_labels = processed_inputs["input_labels"].unsqueeze(0).to(DEVICE)
        
        # 3. Inference
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values, 
                input_points=input_points,
                input_labels=input_labels,
                multimask_output=False
            )
        
        pred_prob = torch.sigmoid(outputs.pred_masks).cpu().numpy().squeeze()
        pred_mask = (pred_prob > 0.5).astype(np.uint8) * 255
        pred_img = Image.fromarray(pred_mask).resize(image_raw.size, Image.NEAREST)
        
        # 4. Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_raw)
        axes[0].set_title(f"Test Input")
        axes[1].imshow(gt_mask_raw, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_img, cmap='gray')
        axes[2].set_title(f"Prediction (IoU Check)")
        
        save_path = os.path.join(vis_dir, f"test_{i}_{os.path.basename(entry['image_path'])}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    model.train()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[*] Loading SAM 2.1 (Transformers)...")
    try:
        processor = Sam2Processor.from_pretrained(MODEL_ID)
        model = Sam2Model.from_pretrained(MODEL_ID).to(DEVICE)
    except Exception as e:
        print(f"[!] Transformers Error: {e}")
        return

    # --- FREEZING STRATEGY ---
    for name, param in model.named_parameters():
        # breakpoint()
        if "mask_decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # --- DATASET & SPLITTING ---
    full_dataset = SAM2Dataset(DATASET_ROOT, processor, mode="bar")
    
    # Split: 90% Train, 10% Test
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"[*] Split: {train_size} Training | {test_size} Testing")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch 1 for accurate IoU calc
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    print("[*] Training Start...")

    best_test_iou = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        optimizer.zero_grad()
        
        # --- TRAINING LOOP ---
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_points = batch["input_points"].to(DEVICE)
            input_labels = batch["input_labels"].to(DEVICE)
            gt_masks = batch["ground_truth_mask"].to(DEVICE).unsqueeze(1) 
            
            outputs = model(
                pixel_values=pixel_values, 
                input_points=input_points,
                input_labels=input_labels,
                multimask_output=False 
            )
            
            pred_masks = outputs.pred_masks.squeeze(1)
            loss = loss_fn(pred_masks, gt_masks)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})
        
        # --- EVALUATION LOOP (Check Overfitting) ---
        print(f"\n[*] Evaluating on {test_size} Unseen Test Images...")
        test_loss, test_iou = evaluate_model(model, test_loader, desc=f"Epoch {epoch+1} [Test]")
        
        print(f"Results Ep {epoch+1}: Train Loss {total_loss/len(train_loader):.4f} | Test Loss {test_loss:.4f} | Test IoU {test_iou:.4f}")
        
        if test_iou > best_test_iou:
            best_test_iou = test_iou
            print(f"  >>> NEW BEST MODEL! Saving...")
            save_path = os.path.join(OUTPUT_DIR, "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
        
        # Save per-epoch checkpoint
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-ep{epoch+1}"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-ep{epoch+1}"))
        
        # Visualize TEST images (Not train images)
        visualize_validation(model, test_dataset, epoch, OUTPUT_DIR)

if __name__ == "__main__":
    main()