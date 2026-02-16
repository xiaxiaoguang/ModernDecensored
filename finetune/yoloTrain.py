import os
import random
import shutil
import yaml
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Source directories (where your current unsplit data is)
SOURCE_IMAGES = "./dataset_refined/yolo/images"

SOURCE_LABELS = "./dataset_refined/yolo/labels"

# Target YOLO structured dataset
DATASET_ROOT = "./dataset_yolo" 
DATA_YAML = os.path.join(DATASET_ROOT, "manga_data.yaml")

MODEL_SIZE = "yolo11l.pt"  
TRAIN_GPUS = [1]
TRAIN_EPOCHS = 100
TRAIN_IMGSZ = 1024

HOME_GPU_ID = 0
# =================================================

def setup_and_split_dataset(train_ratio=0.8):
    """
    Creates YOLO directory structure and randomly splits data into Train/Val sets.
    """
    print(f"--- 🗂️ PREPARING DATASET ({train_ratio*100}% Train / {(1-train_ratio)*100}% Val) ---")
    
    # Create YOLO folder structure
    dirs = {
        "train_img": os.path.join(DATASET_ROOT, "images/train"),
        "val_img": os.path.join(DATASET_ROOT, "images/val"),
        "train_lbl": os.path.join(DATASET_ROOT, "labels/train"),
        "val_lbl": os.path.join(DATASET_ROOT, "labels/val")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Get all images
    all_images = [f for f in os.listdir(SOURCE_IMAGES) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]
    
    def copy_files(file_list, img_dest, lbl_dest):
        valid_count = 0
        for img_name in file_list:
            base_name = os.path.splitext(img_name)[0]
            txt_name = f"{base_name}.txt"
            
            src_img = os.path.join(SOURCE_IMAGES, img_name)
            src_lbl = os.path.join(SOURCE_LABELS, txt_name)
            
            # Only copy if both image and label exist
            if os.path.exists(src_lbl):
                shutil.copy(src_img, os.path.join(img_dest, img_name))
                shutil.copy(src_lbl, os.path.join(lbl_dest, txt_name))
                valid_count += 1
        return valid_count

    print("Copying Training files...")
    train_count = copy_files(train_files, dirs["train_img"], dirs["train_lbl"])
    print("Copying Validation files...")
    val_count = copy_files(val_files, dirs["val_img"], dirs["val_lbl"])
    
    print(f"✅ Split Complete! Train: {train_count} | Val: {val_count}\n")

def create_yaml():
    print("--- 📝 GENERATING DATA.YAML ---")
    
    yaml_content = {
        "path": DATASET_ROOT,
        "train": "images/train",  
        "val": "images/val",    
        "names": {
            0: "mosaic",
            1: "black_bar"
        }
    }
    
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"✅ Created {DATA_YAML}\n")

def train_model():
    print(f"--- 🚀 STARTING TRAINING ON {len(TRAIN_GPUS)} GPUs ---")
    model = YOLO(MODEL_SIZE)

    model.train(
        data=DATA_YAML,
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=10,
        device=TRAIN_GPUS,      
        workers=16,
        project="manga_decensor_project",
        name=MODEL_SIZE,
        exist_ok=True,
        mosaic=1.0, 
        degrees=15.0,
        scale=0.5,              
        fliplr=0.5,             
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0, 
        amp=False,
    )
    print("--- ✅ TRAINING COMPLETE ---")
    return "runs/detect/manga_decensor_project/11mrun/weights/best.pt"

def test_inference_and_mask(weights_path):
    print(f"\n--- 🏠 RUNNING INFERENCE ON VALIDATION SET ---")
    model = YOLO(weights_path)
    val_img_dir = os.path.join(DATASET_ROOT, "images/val")
    output_dir = os.path.join(DATASET_ROOT, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    val_images = [f for f in os.listdir(val_img_dir) if f.endswith(('.png', '.jpg'))]
    if not val_images:
        print("No validation images found for testing.")
        return

    # Test on the first 5 images in the validation set
    for img_name in val_images[:10]:
        img_path = os.path.join(val_img_dir, img_name)
        
        # High Recall Inference (iou=0.6, conf=0.05)
        # imgsz=1024 tells YOLO to process at high-res, but it returns original coordinates!
        results = model.predict(img_path, imgsz=1024, conf=0.05, iou=0.6, device=HOME_GPU_ID)
        
        # Read the raw, unscaled image
        orig_img = cv2.imread(img_path)
        final_mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
        
        found_bars = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            
            if cls_id == 1: # Black bars
                found_bars += 1
                # xyxy[0] returns coordinates perfectly mapped to the original image dimensions
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Crop the raw region
                roi = orig_img[y1:y2, x1:x2]
                # Extract black pixels
                is_black = np.all(roi <= [5, 5, 5], axis=-1)
                final_mask[y1:y2, x1:x2][is_black] = 255
                # Draw green box for visualization
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_debug.jpg"), orig_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_anchor_mask.png"), final_mask)
        
        print(f"Processed {img_name} -> Found {found_bars} bars.")

if __name__ == "__main__":
    # 1. Prepare data
    # setup_and_split_dataset(train_ratio=0.9)
    # create_yaml()
    
    # 2. Train
    best_weights = train_model()
    
    # 3. Test
    test_inference_and_mask(best_weights)