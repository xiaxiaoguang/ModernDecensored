import os
from pathlib import Path

def clean_dataset(root_path):
    subfolders = ['inpainter_bar', 'inpainter_mosaic']
    categories = ['censored', 'ground_truth', 'mask']
    
    for sub in subfolders:
        for cat in categories:
            folder_path = Path(root_path) / sub / cat
            if not folder_path.exists():
                continue
                
            print(f"Cleaning {folder_path}...")
            
            for file_path in folder_path.glob("*.png"):
                filename = file_path.stem
                try:
                    file_id = filename.split('_')[-1]
                    if file_id != "0":
                        file_path.unlink()
                except IndexError:
                    print(f"Skipping malformed filename: {file_path.name}")

def sync_masks(root_path):
    """
    Removes images in censored/ground_truth that do not have a corresponding mask.
    Assumes filenames share the same suffix/ID format.
    """
    subfolders = ['inpainter_bar', 'inpainter_mosaic']
    
    for sub in subfolders:
        base_path = Path(root_path) / sub
        mask_folder = base_path / 'mask'
        
        if not mask_folder.exists():
            print(f"Mask folder missing in {sub}, skipping sync...")
            continue

        # 1. Create a set of all valid filenames (or IDs) present in the mask folder
        # We use a set for O(1) lookup speed
        valid_masks = {f.name for f in mask_folder.glob("*.png")}
        
        print(f"Syncing {sub} based on {len(valid_masks)} masks...")

        # 2. Check censored and ground_truth folders
        for cat in ['censored', 'ground_truth']:
            target_folder = base_path / cat
            if not target_folder.exists():
                continue

            for file_path in target_folder.glob("*.png"):
                # If this specific filename isn't in our mask set, delete it
                if file_path.name not in valid_masks:
                    print(f"Deleting {file_path.name} (No matching mask)")
                    file_path.unlink()

if __name__ == "__main__":
    dataset_dir = "./dataset_refined"
    
    # Run the original cleanup (removing non-zero IDs)
    # clean_dataset(dataset_dir)
    
    # Run the new sync function
    sync_masks(dataset_dir)