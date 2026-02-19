import os
import time
import torch
import gc
import numpy as np
import configparser
import shutil
import argparse
import re

from PIL import Image, ImageFilter, ImageChops, ImageOps
from scipy.ndimage import label, binary_dilation, center_of_mass 

# --- CONFIGURATION DEFAULTS ---
DEFAULT_HAI_PATH = r"E:\MangaTranslator\HAI\HAI\main.exe"
DEFAULT_INPUT_FOLDER = r"E:\MangaTranslator\HAI\HAI\input"
DEFAULT_OUTPUT_FOLDER = r"E:\MangaTranslator\HAI\HAI\output"
DEFAULT_DEBUG_FOLDER = r"E:\MangaTranslator\HAI\HAI\debug_masks"
DEFAULT_TEMP_TILES_FOLDER = r"E:\MangaTranslator\HAI\HAI\temp_tiles"
# SAM2_MODEL_PATH = r"E:\MangaTranslator\MangaTranslator\models\sam\models--facebook--sam2.1-hiera-large\snapshots\665f8e2ad61cf5f53d65644ff27c8ee525124610"
SAM2_MODEL_PATH=r"E:\MangaTranslator\HAI\HAI\sam2_saved\train_model60"
LORA_PATH = r"E:\MangaTranslator\HAI\HAI\lora"
YOLO_PATH = r"E:\MangaTranslator\HAI\HAI\yolo_saved\yolo11l.pt"

# Performance & Quality Settings
MAX_DIMENSION = 2000
MIN_DIMENSION = 2000
LOGICAL_TILE_W = 2000
LOGICAL_TILE_H = 2000
CONTEXT_PADDING = 0
INPAINT_SIZE = 1024
MASK_EXPANSION_PIXELS = 2


# --- SHARED UTILITIES ---

def cleanup_gpu():
    """Clears models from GPU memory."""
    print("正在清理 GPU 显存以加载下一个模型...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(2)

def smart_resize(img):
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        scale = MAX_DIMENSION / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"降采样至: {new_w}x{new_h}")
        return img.resize((new_w, new_h), Image.LANCZOS)
    elif max(w, h) < MIN_DIMENSION:
        scale = MIN_DIMENSION / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"Upscaling (Lanczos) to: {new_w}x{new_h}")
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img

def get_tiles_with_padding(img_size):
    w, h = img_size
    tiles = []
    nx = int(np.ceil(w / LOGICAL_TILE_W))
    ny = int(np.ceil(h / LOGICAL_TILE_H))
    if nx > 0: actual_tile_w = int(np.ceil(w / nx))
    else: actual_tile_w = w
    
    if ny > 0: actual_tile_h = int(np.ceil(h / ny))
    else: actual_tile_h = h

    for iy in range(ny):
        for ix in range(nx):
            x1 = ix * actual_tile_w
            y1 = iy * actual_tile_h
            x2 = min(w, (ix + 1) * actual_tile_w)
            y2 = min(h, (iy + 1) * actual_tile_h)
            
            px1 = x1 - CONTEXT_PADDING
            py1 = y1 - CONTEXT_PADDING
            px2 = x2 + CONTEXT_PADDING
            py2 = y2 + CONTEXT_PADDING
            
            tiles.append({
                'logical': (x1, y1, x2, y2),
                'padded': (px1, py1, px2, py2)
            })
    return tiles

def extract_padded_tile(img, padded_box):
    px1, py1, px2, py2 = padded_box
    target_w = px2 - px1
    target_h = py2 - py1
    
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    img_w, img_h = img.size
    
    overlap_x1 = max(0, px1 + CONTEXT_PADDING)
    overlap_y1 = max(0, py1 + CONTEXT_PADDING)
    overlap_x2 = min(img_w, px2 - CONTEXT_PADDING)
    overlap_y2 = min(img_h, py2 - CONTEXT_PADDING)
    
    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        part = img.crop((overlap_x1, overlap_y1, overlap_x2, overlap_y2))
        canvas.paste(part, (CONTEXT_PADDING // 2, CONTEXT_PADDING // 2))
    return canvas

def extract_mask_from_green(image_with_green):
    img = image_with_green.convert("RGB")
    data = np.array(img)
    mask_arr = ((data[:,:,1] > 220) & (data[:,:,0] < 40) & (data[:,:,2] < 40)).astype(np.uint8) * 255
    return Image.fromarray(mask_arr)
