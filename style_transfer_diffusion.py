import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from fft_style_conditioner import FFTStyleConditioner

class StyleTransferDiffusion:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", device="cuda"):
        self.device = device
        
        # Load pretrained models
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        self.pipeline.to(self.device)
        
        # Extract UNet for direct access
        self.unet = self.pipeline.unet
        
        # FFT Style Conditioner
        self.style_conditioner = FFTStyleConditioner().to(self.device)
        
        # Freeze base model weights
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # Adapter layers for injecting style information
        self.style_adapter = torch.nn.ModuleList([
            torch.nn.Conv2d(512, 320, kernel_size=1),  # For down_blocks[2]
            torch.nn.Conv2d(512, 640, kernel_size=1),  # For down_blocks[3]
            torch.nn.Conv2d(512, 640, kernel_size=1),  # For mid_block
            torch.nn.Conv2d(512, 320, kernel_size=1),  # For up_blocks[0]
            torch.nn.Conv2d(512, 320, kernel_size=1),  # For up_blocks[1]
        ]).to(self.device)
        
        # Store original forward methods to modify
        self.original_down_forwards = []
        self.original_mid_forward = None
        self.original_up_forwards = []
        
        # Style condition storage
        self.current_style_condition = None
        
    def inject_style_conditioning(self):
        """Modify UNet layers to inject style condition"""
        # Save original forward methods
        self.original_down_forwards = [
            self.unet.down_blocks[2].forward,
            self.unet.down_blocks[3].forward
        ]
        self.original_mid_forward = self.unet.mid_block.forward
        self.original_up_forwards = [
            self.unet.up_blocks[0].forward,
            self.unet.up_blocks[1].forward
        ]
        
        # Wrap forward methods to inject style
        def wrap_down_forward(block_idx, original_forward):
            adapter_idx = block_idx - 2  # Start from down_blocks[2]
            
            def wrapped_forward(x, temb=None, **kwargs):
                result = original_forward(x, temb, **kwargs)
                
                if self.current_style_condition is not None:
                    # Adapt style condition to required channels
                    style_features = self.style_adapter[adapter_idx](self.current_style_condition)
                    # Add as residual
                    result = result + style_features
                
                return result
            
            return wrapped_forward
        
        def wrap_mid_forward(original_forward):
            def wrapped_forward(x, temb=None, **kwargs):
                result = original_forward(x, temb, **kwargs)
                
                if self.current_style_condition is not None:
                    # Adapt style condition
                    style_features = self.style_adapter[2](self.current_style_condition)
                    # Add as residual
                    result = result + style_features
                
                return result
            
            return wrapped_forward
        
        def wrap_up_forward(block_idx, original_forward):
            adapter_idx = block_idx + 3  # Continue after mid_block
            
            def wrapped_forward(x, temb=None, **kwargs):
                result = original_forward(x, temb, **kwargs)
                
                if self.current_style_condition is not None:
                    # Adapt style condition
                    style_features = self.style_adapter[adapter_idx](self.current_style_condition)
                    # Add as residual
                    result = result + style_features
                
                return result
            
            return wrapped_forward
        
        # Apply wrapped forward methods
        self.unet.down_blocks[2].forward = wrap_down_forward(2, self.original_down_forwards[0])
        self.unet.down_blocks[3].forward = wrap_down_forward(3, self.original_down_forwards[1])
        self.unet.mid_block.forward = wrap_mid_forward(self.original_mid_forward)
        self.unet.up_blocks[0].forward = wrap_up_forward(0, self.original_up_forwards[0])
        self.unet.up_blocks[1].forward = wrap_up_forward(1, self.original_up_forwards[1])
    
    def restore_original_forwards(self):
        """Restore original forward methods"""
        self.unet.down_blocks[2].forward = self.original_down_forwards[0]
        self.unet.down_blocks[3].forward = self.original_down_forwards[1]
        self.unet.mid_block.forward = self.original_mid_forward
        self.unet.up_blocks[0].forward = self.original_up_forwards[0]
        self.unet.up_blocks[1].forward = self.original_up_forwards[1]
    
    def generate_stylized_image(self, content_img, style_img, 
                               num_inference_steps=50, 
                               guidance_scale=7.5, 
                               content_strength=0.8):
        """Generate stylized image using diffusion model with FFT-based style conditioning"""
        # Process images to get style condition
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)
        
        # Extract style conditioning using FFT
        self.current_style_condition = self.style_conditioner(content_img, style_img)
        
        # Inject style conditioning into model
        self.inject_style_conditioning()
        
        # Create empty prompt embedding (we're using image conditioning instead)
        empty_prompt = ""
        prompt_embeds = self.pipeline._encode_prompt(
            empty_prompt, 
            device=self.device, 
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0
        )
        
        # Create latent from content image
        with torch.no_grad():
            content_latent = self.pipeline.vae.encode(content_img * 2.0 - 1.0).latent_dist.sample()
            content_latent = content_latent * self.pipeline.vae.config.scaling_factor
        
        # Run denoising with style conditioning
        generator = torch.Generator(device=self.device).manual_seed(42)  # For reproducibility
        
        # Generate image
        images = self.pipeline(
            prompt_embeds=prompt_embeds,
            latents=content_latent,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            strength=1.0 - content_strength  # How much to preserve content
        ).images
        
        # Restore original model
        self.restore_original_forwards()
        self.current_style_condition = None
        
        return images