from PIL import Image
import torch

from style_transfer_diffusion import StyleTransferDiffusion
from dataset import transform

def generate_stylized_images(content_path, style_path, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load images
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    
    # Apply transformations
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)
    
    # Initialize model
    model = StyleTransferDiffusion(model_id="CompVis/stable-diffusion-v1-4", device=device)
    
    # Load trained weights
    checkpoint = torch.load(model_path)
    model.style_conditioner.load_state_dict(checkpoint['style_conditioner'])
    model.style_adapter.load_state_dict(checkpoint['style_adapter'])
    
    # Generate stylized image
    stylized_images = model.generate_stylized_image(
        content_tensor, 
        style_tensor,
        num_inference_steps=50,
        guidance_scale=7.5,
        content_strength=0.8
    )
    
    # Save output
    stylized_img = stylized_images[0]
    stylized_img.save(output_path)
    
    return stylized_img