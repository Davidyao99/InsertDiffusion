from PIL import Image
import numpy as np
from math import ceil


# get a mask image from an input image using a threshold all pixels that exceed the threshold are included in the mask, all others excluded
def get_mask_from_image(image: Image, threshold: int=100) -> Image:
    fn = lambda x : 255 if x > threshold else 0
    mask_img = image.convert('L').point(fn, mode='1')
    return mask_img

# Improved crop_out function that returns the bounding box
def crop_out(im: Image, threshold: int=245, bounding_box=None):
    """
    Crop the image based on non-white pixels or a provided bounding box.
    
    Args:
        im: Image to crop
        threshold: Threshold for considering pixels as part of the object
        bounding_box: Optional tuple (left, top, right, bottom) for cropping
        
    Returns:
        Cropped image and the bounding box used
    """
    if bounding_box is not None:
        # Use the provided bounding box for cropping
        left_most, top_most, right_most, bottom_most = bounding_box
    else:
        arr = np.array(im)
        # Handle differently based on image mode
        if len(arr.shape) == 3:  # RGB image
            # find all indices in the image that have values below the threshold (not white)
            indices = np.where(arr < threshold)
        else:  # Grayscale/mask image
            # find all indices where the mask is not background (not black)
            indices = np.where(arr > 0)
            
        if len(indices[0]) == 0:  # No pixels found
            # Return original image if no object detected
            return im, (0, 0, im.width, im.height)
            
        # determine minmal and maximal indices along each dimension
        try:
            left_most = ceil(np.min(indices[1])/2)*2
            right_most = ceil(np.max(indices[1])/2)*2
            top_most = ceil(np.min(indices[0])/2)*2
            bottom_most = ceil(np.max(indices[0])/2)*2
        except ValueError:
            print(f"Warning: Could not find object in image. Using full image.")
            return im, (0, 0, im.width, im.height)

    # ensure proper box
    if right_most - left_most <= 0:
        right_most = left_most + 2
    if bottom_most - top_most <= 0:
        bottom_most = top_most + 2

    # crop image using the bounding box
    cropped = im.crop((left_most, top_most, right_most, bottom_most))
    return cropped, (left_most, top_most, right_most, bottom_most)

# scaling of object
def scale_img(cropped: Image, scaling_factor: float) -> Image:
    return cropped.resize((ceil((cropped.width*scaling_factor)/2)*2, ceil((cropped.height*scaling_factor)/2)*2), Image.Resampling.NEAREST)

# Modified paste function without the breakpoint
def paste(cropped: Image, original_image: Image, fraction_down: float, fraction_right: float, rescale: bool=False, is_mask: bool=False):
    # calculate new starting location of object
    down_pixels = ceil(fraction_down * original_image.height)
    right_pixels = ceil(fraction_right * original_image.width)

    # ensure object fits at position
    if down_pixels < cropped.height//2:
        raise AssertionError('Your object is too close to the top to fit, either downscale or move down')
    if original_image.height - (original_image.height - down_pixels) < cropped.height//2:
        raise AssertionError('Your object is too close to the bottom to fit, either downscale or move up')
    if right_pixels < cropped.width//2:
        raise AssertionError('Your object is too close to the left to fit, either downscale or move right')
    if original_image.width - (original_image.width - right_pixels) < cropped.width//2:
        raise AssertionError('Your object is too close to the right to fit, either downscale or move left')

    cropped_arr = np.array(cropped)
    cropped_height, cropped_width = cropped_arr.shape[0], cropped_arr.shape[1]
    
    # Create the appropriate canvas based on whether it's a mask or image
    if is_mask:
        if len(np.array(original_image).shape) == 3:  # RGB image
            empty = np.zeros((original_image.height, original_image.width), dtype=np.uint8)  # Create pure black grayscale image
        else:
            empty = np.zeros_like(np.array(original_image), dtype=np.uint8)  # Create pure black image matching original
    else:
        empty = (np.ones_like(np.array(original_image))*255).astype(np.uint8)  # Create pure white image

    # Calculate coordinates for pasting
    start_y = down_pixels - cropped_height//2
    end_y = start_y + cropped_height
    start_x = right_pixels - cropped_width//2
    end_x = start_x + cropped_width
    
    # Insert object at new location
    if len(empty.shape) == len(cropped_arr.shape):
        empty[start_y:end_y, start_x:end_x] = cropped_arr
    elif len(empty.shape) == 2 and len(cropped_arr.shape) == 3:  # Pasting RGB onto grayscale
        empty[start_y:end_y, start_x:end_x] = np.mean(cropped_arr, axis=2).astype(np.uint8)
    elif len(empty.shape) == 3 and len(cropped_arr.shape) == 2:  # Pasting grayscale onto RGB
        for i in range(empty.shape[2]):
            empty[start_y:end_y, start_x:end_x, i] = cropped_arr
    
    image = Image.fromarray(empty)
    
    # optionally rescale image to have width and height 256
    if rescale:
        mode = 'L' if is_mask else 'RGB'
        bg_color = 0 if is_mask else 255
        
        if image.width > image.height:
            empty_im = Image.new(mode, (image.width, image.width), color=bg_color)
        else:
            empty_im = Image.new(mode, (image.height, image.height), color=bg_color)
            
        empty_im.paste(image, (empty_im.width//2-image.width//2, empty_im.height//2-image.height//2))
        image = empty_im.resize((256, 256), Image.Resampling.BICUBIC)

    return image

# paste_pipeline repositions the object in its frame using the same operations for both image and mask
def paste_pipeline(im: Image, scale: float=1, fraction_down: float=0.5, fraction_right: float=0.5, rescale: bool=False, rotation: float=0, mask: Image = None) -> Image:
    """
    Apply identical transformations (crop, scale, paste, rotate) to both image and mask.
    """
    # If no mask is provided, just process the image
    if mask is None:
        # First, crop the image and get bounding box
        cropped_img, _ = crop_out(im)
        # Then scale, paste, and rotate
        scaled_img = scale_img(cropped_img, scale)
        pasted_img = paste(scaled_img, im, fraction_down, fraction_right, rescale)
        rotated_img = pasted_img.rotate(rotation, fillcolor='white')
        return rotated_img
    
    # If a mask is provided, use the same bounding box for both
    # First find the bounding box from the image
    _, bbox = crop_out(im)
    
    # Apply crop using the same bbox to both image and mask
    cropped_img, _ = crop_out(im, bounding_box=bbox)
    cropped_mask, _ = crop_out(mask, bounding_box=bbox)
    
    # Apply the same scaling to both
    scaled_img = scale_img(cropped_img, scale)
    scaled_mask = scale_img(cropped_mask, scale)
    
    # Paste both with the same parameters
    pasted_img = paste(scaled_img, im, fraction_down, fraction_right, rescale)
    pasted_mask = paste(scaled_mask, mask, fraction_down, fraction_right, rescale, is_mask=True)
    
    # Rotate both with appropriate fill colors
    rotated_img = pasted_img.rotate(rotation, fillcolor='white')
    rotated_mask = pasted_mask.rotate(rotation, fillcolor=0)
    
    return rotated_img, rotated_mask