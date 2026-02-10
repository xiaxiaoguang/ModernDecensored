import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Training Settings (For your 4x 3090 Server)
DATA_YAML = "G:/path/to/dataset_yolo/data.yaml"
MODEL_SIZE = "yolo11x.pt"  # The largest, smartest model
TRAIN_GPUS = [0, 1, 2, 3]  # Use all 4 GPUs
TRAIN_EPOCHS = 100
TRAIN_IMGSZ = 1024         # High Res for small black bars

# Inference Settings (Simulating Home RTX 3070)
TEST_IMAGE_PATH = "test_manga_page.jpg" # Put a sample image here
HOME_GPU_ID = 0            # Simulate single GPU inference
# =================================================

def train_model():
    print(f"\n--- 🚀 STARTING TRAINING ON {len(TRAIN_GPUS)} GPUs ---")
    print(f"Model: {MODEL_SIZE} | Resolution: {TRAIN_IMGSZ}px")
    
    # Load Pre-trained weights
    model = YOLO(MODEL_SIZE)

    # Train
    model.train(
        data=DATA_YAML,
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMGSZ,
        batch=64,               # 4x3090s can handle 64-128 easily
        device=TRAIN_GPUS,      # Parallel GPU training
        workers=16,
        project="manga_decensor_project",
        name="final_run",
        exist_ok=True,
        
        # Augmentations to help learn bars
        mosaic=1.0, 
        degrees=15.0,           # Slight rotation helps learn angled bars
        scale=0.5,              # Helps learn bars at different zoom levels
    )
    print("--- ✅ TRAINING COMPLETE ---")
    return "manga_decensor_project/final_run/weights/best.pt"

def test_inference_speed(weights_path):
    print(f"\n--- 🏠 SIMULATING HOME USER (RTX 3070 Level) ---")
    
    # Force usage of only 1 GPU to simulate home environment
    device = torch.device(f"cuda:{HOME_GPU_ID}")
    
    # Load the fine-tuned model
    print(f"Loading weights: {weights_path}")
    model = YOLO(weights_path)
    
    # Create a dummy image if file doesn't exist (1024x1024)
    if not os.path.exists(TEST_IMAGE_PATH):
        dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
        cv2.imwrite(TEST_IMAGE_PATH, dummy)

    # 1. Check VRAM Usage
    # Reset max memory tracker
    torch.cuda.reset_peak_memory_stats(device)
    
    # 2. Warmup (Compile model graph)
    print("Warming up...")
    for _ in range(10):
        model.predict(TEST_IMAGE_PATH, imgsz=1024, verbose=False, device=HOME_GPU_ID)
        
    # 3. Speed Benchmark
    print("Running Benchmark (100 iterations)...")
    t0 = time.time()
    for _ in range(100):
        results = model.predict(TEST_IMAGE_PATH, imgsz=1024, verbose=False, device=HOME_GPU_ID)
    t1 = time.time()
    
    total_time = t1 - t0
    avg_latency = (total_time / 100) * 1000 # ms
    fps = 100 / total_time
    
    # Get Peak VRAM used
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 # MB
    
    print(f"\n" + "="*40)
    print(f"   INFERENCE RESULTS (1024px)")
    print(f"="*40)
    print(f"   Model:         YOLO11x (Extra Large)")
    print(f"   Average Speed: {avg_latency:.2f} ms per image")
    print(f"   FPS:           {fps:.2f} FPS")
    print(f"   VRAM Used:     {peak_mem:.2f} MB")
    print(f"="*40)
    
    if peak_mem < 6000:
        print("✅ SUCCESS: Fits comfortably on RTX 3070 (8GB)")
    else:
        print("⚠️ WARNING: Might be tight on 8GB cards")

if __name__ == "__main__":
    import os
    
    # 1. Run Training
    best_weights = train_model()
    
    # 2. Run Inference Test
    test_inference_speed(best_weights)