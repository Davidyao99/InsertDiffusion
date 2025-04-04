from .plotting_utils import concat_PIL_h, concat_PIL_v
from .mask import get_mask_from_image, paste_pipeline
from .stable_diffusion import sd_inpainting, sd_img2img, sd_colorization
from .load_bikes import get_bikes_and_masks
from .bike_diffusion.get_mask import get_mask_and_background, get_dataframe_row
from .prompt import get_bike_inpainting_prompt, get_bike_colorization_prompt, get_bike_prompts, get_car_inpainting_prompt, get_car_colorization_prompt, get_car_prompts, get_product_inpainting_prompt, get_product_colorization_prompt, get_product_prompts
from .bike_diffusion.repaint_diffusion import inpaint, inpaint_tensor_to_image
from .bike_diffusion.diffusion import get_diffusion_runner, load_model, get_args, get_config
from .langsam.langsam_masks import get_mask_and_bb_langsam, get_pasted_image
from .extraction import paste_image
from PIL import Image

def extract_object(image, background, object_desc):
    """
    Extract an object from an image based on description and paste it onto a white background.
    Returns both the extracted image and the extracted object's mask.
    """
    # Use default erosion strength of 0
    extracted_image, object_mask = get_pasted_image(image, object_desc, erosion_strength=0)
    
    # If background is different from the image, ensure the extracted image has the same dimensions
    if background.size != image.size:
        # Create a white canvas with background dimensions
        canvas = Image.new('RGB', background.size, (255, 255, 255))
        # Paste the extracted image centered on the canvas
        paste_x = (canvas.width - extracted_image.width) // 2
        paste_y = (canvas.height - extracted_image.height) // 2
        canvas.paste(extracted_image, (paste_x, paste_y))
        
        # Also adjust the mask to match the background dimensions
        mask_canvas = Image.new('L', background.size, 0)  # Black canvas for mask (0)
        mask_canvas.paste(object_mask, (paste_x, paste_y))
        
        return canvas, mask_canvas
    
    return extracted_image, object_mask
