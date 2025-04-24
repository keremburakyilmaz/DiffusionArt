import os
import gradio as gr
import torch
from PIL import Image
import numpy as np

from style_transfer_diffusion import StyleTransferDiffusion
from dataset import transform
import config

def load_model(checkpoint_path):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StyleTransferDiffusion(model_id=config.DIFFUSION_MODEL_ID, device=device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.style_conditioner.load_state_dict(checkpoint['style_conditioner'])
        model.style_adapter.load_state_dict(checkpoint['style_adapter'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    return model

def stylize_image(content_img, style_img, content_strength, guidance_scale, num_steps):
    """Stylize an image using the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load latest checkpoint
    checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pth')]
    if not checkpoints:
        return None, "No checkpoints found. Please train the model first."
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, latest_checkpoint)
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Preprocess images
    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)
    
    # Generate stylized image
    stylized_images = model.generate_stylized_image(
        content_tensor,
        style_tensor,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        content_strength=float(content_strength)
    )
    
    return stylized_images[0], f"Stylized with checkpoint: {latest_checkpoint}"

def create_webui():
    """Create a Gradio interface for the style transfer model"""
    with gr.Blocks(title="FFT Style Transfer") as app:
        gr.Markdown("# FFT-based Style Transfer with Diffusion Models")
        
        with gr.Row():
            with gr.Column(scale=1):
                content_image = gr.Image(label="Content Image", type="pil")
                style_image = gr.Image(label="Style Image", type="pil")
                
                with gr.Row():
                    content_strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.05, 
                                               label="Content Strength")
                    guidance_scale = gr.Slider(minimum=1.0, maximum=12.0, value=7.5, step=0.5, 
                                             label="Guidance Scale")
                
                num_steps = gr.Slider(minimum=10, maximum=100, value=50, step=5, 
                                     label="Diffusion Steps")
                
                generate_btn = gr.Button("Generate Stylized Image")
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Stylized Output")
                output_info = gr.Textbox(label="Information")
        
        generate_btn.click(
            fn=stylize_image,
            inputs=[content_image, style_image, content_strength, guidance_scale, num_steps],
            outputs=[output_image, output_info]
        )
    
    return app

# Launch the app
if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    app = create_webui()
    app.launch(share=True)