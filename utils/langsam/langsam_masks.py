import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import requests
from PIL import Image
from io import BytesIO
import os
import sys
from scipy.ndimage import binary_erosion

parent_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the system path
sys.path.append(parent_dir)

from lang_sam import LangSAM
import torch

# in this file we use the lang_sam package by Luca Medeiros (https://github.com/luca-medeiros/lang-segment-anything) to extract objects from the original background and insert them into a clean white background

# model as a global variable s.t. it is not loaded new everytime the functions in the file are used
model = None

# function to visualize the outputs including image, box, mask, and prompt
def visualize_langsam_output(image, mask, box, prompt, save_path=None):
    # Convert image to numpy if it's a PIL Image
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # Display the image
    ax.imshow(img_np)
    
    # Create a semi-transparent colormap for the mask
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.4)]  # Transparent to red with alpha
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    # Display the mask if available
    if mask is not None:
        ax.imshow(mask, cmap=custom_cmap)
    
    # Draw bounding box if available
    if box is not None:
        # Box format is [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add prompt text near the box
        ax.text(x1, y1-10, prompt, color='white', fontsize=12, 
                bbox=dict(facecolor='red', alpha=0.7, pad=3))
    
    plt.title(f"Object Detection: '{prompt}'")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.savefig("preprocess.png", bbox_inches='tight')
    
# this method is used to generate segmentation masks and bounding boxes using a prompt
def get_mask_and_bb_langsam(image: Image, prompt: str, visualize=True):
    global model
    if model is None:
        model = LangSAM()
    # use langSAM to predict all bounding boxes and segmentation masks for a given prompt
    masks, boxes, phrases, logits = model.predict(image, prompt)

    # if no object is found return None None
    if len(masks) == 0:
        if visualize:
            visualize_langsam_output(image, None, None, prompt)
        return None, None
    
    # if only a single mask is found the mask is returned
    if len(masks) == 1:
        mask = masks.detach().squeeze().cpu().numpy()
        box = boxes.detach().squeeze().cpu().numpy()
        if visualize:
            visualize_langsam_output(image, mask, box, prompt)

        return mask, box

    # old code to combine all found masks into a single mask:
    # combined_mask = torch.any(masks, dim=0)
    # return combined_mask.detach().cpu().numpy(), boxes[0].detach().cpu().numpy()

    # if more than one mask is found the best fitting mask is returned (maximal logit indicates highest probability)
    max_logit_idx = torch.argmax(logits).item()
    mask = masks[max_logit_idx].detach().cpu().numpy()
    box = boxes[max_logit_idx].detach().cpu().numpy()
    
    if visualize:
        visualize_langsam_output(image, mask, box, prompt)
    
    return mask, box

# this function extracts an object from an image based on the objects description (prompt) and pastes it onto a white image
def get_pasted_image(img: Image, prompt: str, erosion_strength: int):
    # get segmentation mask of object
    mask, _ = get_mask_and_bb_langsam(img, prompt)
    if mask is None:
        # if no fitting object is found an empty mask is created
        print('Object not found!')
        mask = np.zeros_like(np.array(img)).astype(bool)
    if erosion_strength > 0:
        # if a mask is found it is eroded to produce a cleaner mask
        mask = binary_erosion(mask, iterations=erosion_strength)
    im_array = np.array(img)
    # create a pure black image
    im_pasted = np.ones_like(im_array)*255
    # replace masked values to insert object onto background at original position
    im_pasted[mask] = im_array[mask]
    
    # Create a binary mask image suitable for SD inpainting (255 where object is, 0 elsewhere)
    binary_mask = np.zeros((im_array.shape[0], im_array.shape[1]), dtype=np.uint8)
    binary_mask[mask] = 255
    mask_img = Image.fromarray(binary_mask)
    
    return Image.fromarray(im_pasted), mask_img
