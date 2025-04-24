import torch
import os
from tqdm import tqdm
from utility import compute_gram_matrices, extract_vgg_features
from visualization import save_samples, visualize_frequency_bands, plot_losses
import config
from dataset import transform


def train_style_transfer_model(content_loader, style_loader, model, num_epochs=config.NUM_EPOCHS):
    # Optimizer (only train the style conditioner and adapter layers)
    optimizer = torch.optim.Adam([
        {'params': model.style_conditioner.parameters()},
        {'params': model.style_adapter.parameters()}
    ], lr=config.LEARNING_RATE)
    
    # Loss functions
    content_criterion = torch.nn.MSELoss()
    style_criterion = torch.nn.MSELoss()
    
    # For tracking losses
    losses = {
        "content_loss": [],
        "style_loss": [],
        "total_loss": []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        content_loss_sum = 0
        style_loss_sum = 0
        batch_count = 0
        
        # Create paired batches from content and style loaders with progress bar
        for batch_idx, ((content_batch, content_paths), (style_batch, _, style_paths)) in enumerate(
                tqdm(zip(content_loader, style_loader), 
                     desc=f"Epoch {epoch+1}/{num_epochs}", 
                     total=min(len(content_loader), len(style_loader)))
            ):
            
            content_batch = content_batch.to(model.device)
            style_batch = style_batch.to(model.device)
            
            batch_size = content_batch.size(0)
            batch_count += batch_size
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Extract frequency bands for visualization (before passing through model)
            if batch_idx == 0 and epoch % 5 == 0:
                with torch.no_grad():
                    low_freq_c, mid_freq_c, high_freq_c = model.style_conditioner.extract_frequency_bands(content_batch)
                    low_freq_s, mid_freq_s, high_freq_s = model.style_conditioner.extract_frequency_bands(style_batch)
                    visualize_frequency_bands(
                        low_freq_c, mid_freq_c, high_freq_c,
                        config.SAMPLES_DIR, epoch, prefix="content"
                    )
                    visualize_frequency_bands(
                        low_freq_s, mid_freq_s, high_freq_s,
                        config.SAMPLES_DIR, epoch, prefix="style"
                    )
            
            # Generate stylized images
            stylized_images = model.generate_stylized_image(
                content_batch, 
                style_batch,
                num_inference_steps=config.NUM_INFERENCE_STEPS_TRAINING,
                guidance_scale=config.GUIDANCE_SCALE,
                content_strength=config.CONTENT_STRENGTH
            )
            
            # Convert PIL images to tensor
            stylized_tensor = torch.stack([transform(img) for img in stylized_images]).to(model.device)
            
            # Compute losses
            # Content loss (perceptual features from VGG)
            content_features = extract_vgg_features(content_batch)
            stylized_features = extract_vgg_features(stylized_tensor)
            content_loss = content_criterion(content_features[-1], stylized_features[-1])
            
            # Style loss (Gram matrices from VGG)
            style_features = extract_vgg_features(style_batch)
            style_gram_matrices = compute_gram_matrices(style_features)
            stylized_gram_matrices = compute_gram_matrices(stylized_features)
            
            style_loss = 0
            for i in range(len(style_gram_matrices)):
                style_loss += style_criterion(style_gram_matrices[i], stylized_gram_matrices[i])
            style_loss /= len(style_gram_matrices)
            
            # Total loss
            loss = content_loss + 10.0 * style_loss  # Style loss is usually weighted more
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track losses
            total_loss += loss.item() * batch_size
            content_loss_sum += content_loss.item() * batch_size
            style_loss_sum += style_loss.item() * batch_size
            
            # Record losses
            losses["content_loss"].append(content_loss.item())
            losses["style_loss"].append(style_loss.item())
            losses["total_loss"].append(loss.item())
            
            # Save sample images periodically
            if batch_idx % 50 == 0:
                save_samples(
                    content_batch[0], 
                    style_batch[0], 
                    stylized_tensor[0],
                    epoch, 
                    batch_idx,
                    config.SAMPLES_DIR
                )
        
        # Print epoch results
        avg_loss = total_loss / batch_count
        avg_content_loss = content_loss_sum / batch_count
        avg_style_loss = style_loss_sum / batch_count
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Content Loss: {avg_content_loss:.4f}, Style Loss: {avg_style_loss:.4f}")
        
        # Plot losses
        plot_losses(losses, config.OUTPUT_DIR)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"style_transfer_checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'style_conditioner': model.style_conditioner.state_dict(),
                'style_adapter': model.style_adapter.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    return losses