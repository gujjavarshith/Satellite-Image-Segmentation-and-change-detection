import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from method import obtain_change_map, apply_threshold
from metrics import split_neighborhood_uniform

def main():
    # Paths
    # Assuming we are running from the SiROC directory
    workspace_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    print("Please enter the path to the folder containing the images (e.g., 'Dataset - 2/New/pair' or full path).")
    print(f"Dataset - 2 relative path: Dataset - 2/New/pair")
    print(f"Onera Dataset relative path: Onera Dataset/Images/New/pair")
    
    user_input = input("Enter path: ").strip()
    
    # If user provides a relative path, try to resolve it relative to workspace_root
    if not os.path.isabs(user_input):
        potential_path = os.path.join(workspace_root, user_input)
        if os.path.exists(potential_path):
            new_images_path = potential_path
        else:
            # Fallback to treating it as relative to CWD or just taking it as is
            new_images_path = os.path.abspath(user_input)
    else:
        new_images_path = user_input

    out_dir = os.path.join(os.getcwd(), 'Plots')
    
    # Verify setup
    print(f"Checking for images in: {new_images_path}")
    
    img1_path = None
    img2_path = None

    # Helper to find images in a directory
    def find_images(directory):
        if not os.path.exists(directory):
            return None, None
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if len(files) >= 2:
            files.sort()
            return os.path.join(directory, files[0]), os.path.join(directory, files[1])
        return None, None

    # Check pair/ folder
    img1_path, img2_path = find_images(new_images_path)

    # Check New/ folder directly if not found
    if img1_path is None:
        new_images_root = os.path.dirname(new_images_path)
        print(f"Checking for images in: {new_images_root}")
        img1_path, img2_path = find_images(new_images_root)

    if img1_path is None:
        print(f"Error: Images not found in {new_images_path} or {os.path.dirname(new_images_path)}")
        print("Please ensure at least two image files exist in 'Dataset - 2/New/pair/' or 'Dataset - 2/New/'")
        return

    print(f"Found files: {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")

    # Load images
    print(f"Loading images...")
    pre_img_raw = cv2.imread(img1_path)
    post_img_raw = cv2.imread(img2_path)
    
    if pre_img_raw is None:
        print(f"Failed to load {img1_path}")
        return
    if post_img_raw is None:
        print(f"Failed to load {img2_path}")
        return

    pre_img_raw = cv2.cvtColor(pre_img_raw, cv2.COLOR_BGR2RGB)
    post_img_raw = cv2.cvtColor(post_img_raw, cv2.COLOR_BGR2RGB)

    # Convert to torch tensor
    pre_img = torch.from_numpy(pre_img_raw.transpose((2, 0, 1))).float() / 255.0
    post_img = torch.from_numpy(post_img_raw.transpose((2, 0, 1))).float() / 255.0

    # Batch dimension
    pre_img = pre_img.unsqueeze(0)
    post_img = post_img.unsqueeze(0)

    # Parameters
    ensemble = True
    max_neighborhood = 200
    exclusion = 0
    threshold = 'Otsu'
    splits = 27
    otsu_factor = 1.4
    voting_threshold = 0.65

    # Run SiROC
    print("Running SiROC detection...")
    if ensemble:
        neighborhood, ex = split_neighborhood_uniform(max_neighborhood, splits, exclusion)[0]
        change_map = obtain_change_map(pre_img, post_img, neighborhood=neighborhood, excluded=ex)

        for neighborhood, ex in split_neighborhood_uniform(max_neighborhood, splits, exclusion)[1:]:
            change_map = torch.cat((change_map, obtain_change_map(pre_img, post_img, neighborhood=neighborhood, excluded=ex)), dim=0)

        # Take absolute value of change signal
        change_map = torch.abs(change_map)

        # Average across spectral bands
        change_map = change_map.mean(dim=1)

        # Apply threshold for each NN individually
        l = 0
        for j in change_map:
            apply_threshold(change_map, j, threshold, l, otsu_factor)
            l += 1

    # Average the ensemble predictions
    full_map = torch.mean(change_map.float(), dim=0)    
    binary_map = torch.where(full_map >= voting_threshold, torch.tensor(1), torch.tensor(0))
    
    # Save output
    os.makedirs(os.path.join(out_dir, 'Change_Maps'), exist_ok=True)
    
    # Generate a meaningful name for the output files based on the input directory
    folder_name = os.path.basename(os.path.normpath(new_images_path))
    if folder_name.lower() == 'pair':
        # If the folder is 'pair', use the parent directory name (e.g., 'paris' or 'New')
        parent_name = os.path.basename(os.path.dirname(os.path.normpath(new_images_path)))
        # If parent is 'New', try to include grandparent for more context (e.g., 'Dataset-2_New')
        if parent_name.lower() == 'new':
             grandparent_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(new_images_path))))
             run_name = f"{grandparent_name}_{parent_name}"
        else:
             run_name = parent_name
    else:
        run_name = folder_name
        
    # Sanitize filename
    run_name = "".join([c for c in run_name if c.isalnum() or c in ('-', '_')]).strip()

    # Save Heatmap
    heatmap_file = os.path.join(out_dir, 'Change_Maps', f'{run_name}_heatmap.png')
    plt.imsave(heatmap_file, full_map.numpy(), cmap='jet')
    print(f"Heatmap saved to {heatmap_file}")

    # Save Change Map
    out_file = os.path.join(out_dir, 'Change_Maps', f'{run_name}_change_map.png')
    plt.imsave(out_file, binary_map.numpy(), cmap='gray')
    print(f"Change map saved to {out_file}")

if __name__ == "__main__":
    main()
