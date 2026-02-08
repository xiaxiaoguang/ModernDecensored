#!/usr/bin/env python3
"""
webptopdf_multi.py
Modes:
1. Sequential (Default): Merges folders one after another.
2. Index-based (--mode index): Uses numbers in filenames to replace base images.
   Example: folder2/(_006_decensored.png) replaces folder1/0006.png.
"""

import re
import sys
import os
import shutil
import argparse
from pathlib import Path
from PIL import Image

def get_index_from_filename(filename):
    """
    Extracts the first number found in a filename.
    Matches '006' in '(_006_decensored.png)' or '0001' in '0001.jpg'.
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def get_sorted_image_files(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Warning: Folder not found: {folder}")
        return []
    exts = ("*.webp", "*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
        # files.extend(folder.glob(ext.upper()))
    # Sort by name for consistency
    # return sorted(list(set(files)), key=lambda p: (p.name.lower()))
    return files

def process_image(file_path):
    """Loads and converts image to RGB."""
    try:
        img = Image.open(file_path)
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P": img = img.convert("RGBA")
            mask = img.split()[-1] if "A" in img.getbands() else None
            bg.paste(img, mask=mask)
            return bg
        else:
            return img.convert("RGB")
    except Exception as e:
        print(f"    Failed to load {file_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Merge folders to PDF and rename images.")
    parser.add_argument("folders", nargs="+", help="Input folders")
    parser.add_argument("--mode", choices=["seq", "index"], default="seq", 
                        help="seq: merge folders sequentially; index: replace images by numeric index")
    args = parser.parse_args()

    input_folders = [Path(f).resolve() for f in args.folders]
    first_folder = input_folders[0]
    home_dir = first_folder.parent

    # --- ARCHIVE NAME LOGIC ---
    sibling_name = first_folder.name.replace("-translated", "").replace("-t", "")
    sibling_folder = home_dir / sibling_name
    archive_name = next((item.stem for item in sibling_folder.iterdir() 
                        if item.suffix.lower() in {'.zip', '.7z', '.rar'}), sibling_name + "_")
    
    target_path = home_dir / archive_name
    pdf_output = home_dir / "translated" / f"{archive_name}.pdf"

    final_files_map = {} # {index: Path}

    if args.mode == "index":
        print(f"Running in INDEX mode. Folders will overwrite based on numeric IDs.")
        for folder in input_folders:
            files = get_sorted_image_files(folder)
            for f in files:
                idx = get_index_from_filename(f.name)
                if idx is not None:
                    final_files_map[idx] = f
        
        # Sort by the extracted index
        sorted_indices = sorted(final_files_map.keys())
        final_file_list = [final_files_map[i] for i in sorted_indices]
    else:
        print(f"Running in SEQUENTIAL mode.")
        final_file_list = []
        for folder in input_folders:
            final_file_list.extend(get_sorted_image_files(folder))

    if not final_file_list:
        print("No images found.")
        return

    # --- SAVE PDF ---
    print(f"Collecting {len(final_file_list)} images for PDF...")
    pil_images = []
    for f in final_file_list:
        img = process_image(f)
        if img: pil_images.append(img)

    if pil_images:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        pil_images[0].save(pdf_output, save_all=True, append_images=pil_images[1:], 
                           format="PDF", quality=95, dpi=(300, 300))
        print(f"PDF saved: {pdf_output}")

    # --- COPY AND RENAME ---
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Copying files to {target_path}...")
    for count, original_path in enumerate(final_file_list):
        new_name = f"{count:04d}{original_path.suffix.lower()}"
        shutil.copy2(original_path, target_path / new_name)
    
    print(f"Done! Cleaned images saved in: {target_path.name}")

if __name__ == "__main__":
    main()