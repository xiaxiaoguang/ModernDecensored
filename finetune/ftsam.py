import os
import torch
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import AdamW
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.datasets import SAM2Dataset

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
LR = 1e-6
EPOCHS = 200

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
        if "mask_decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # --- DATASET & SPLITTING ---
    full_dataset = SAM2Dataset(DATASET_ROOT, processor, mode="bar")
    
    # Split: 90% Train, 10% Validation
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"[*] Split: {train_size} Training | {val_size} Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) 
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    print("[*] Training Start...")

    best_val_iou = 0.0
    
    # --- METRICS TRACKING ---
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_iou": []
    }

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
        
        avg_train_loss = total_loss / len(train_loader)

        # --- EVALUATION LOOP ---
        print(f"\n[*] Evaluating on {val_size} Unseen Validation Images...")
        # Note: Ensure your evaluate_model function returns (loss, iou)
        val_loss, val_iou = evaluate_model(model, val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        print(f"Results Ep {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f} | Val IoU {val_iou:.4f}")
        
        # --- UPDATE HISTORY & SAVE GRAPHS ---
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        
        save_learning_curve(history, OUTPUT_DIR)
        save_training_log(history, OUTPUT_DIR)
        
        # --- CHECKPOINTING ---
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print(f"  >>> NEW BEST MODEL! Saving...")
            save_path = os.path.join(OUTPUT_DIR, "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
        
        # Save per-epoch checkpoint
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-ep{epoch+1}"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-ep{epoch+1}"))
        
        # Visualize Validation images
        visualize_validation(model, val_dataset, epoch, OUTPUT_DIR)

# --- HELPER FUNCTIONS FOR VISUALIZATION ---

def save_learning_curve(history, output_dir):
    """Generates and updates the learning curve plot after each epoch."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label='Train Loss', marker='o')
    plt.plot(history["epoch"], history["val_loss"], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: IoU Score
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_iou"], label='Validation IoU', color='green', marker='o')
    plt.title('Validation IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"))
    plt.close()

def save_training_log(history, output_dir):
    """Saves the raw metrics to a CSV file for easy inspection later."""
    csv_path = os.path.join(output_dir, "training_log.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation IoU"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i], 
                f"{history['train_loss'][i]:.6f}", 
                f"{history['val_loss'][i]:.6f}", 
                f"{history['val_iou'][i]:.6f}"
            ])

if __name__ == "__main__":
    main()