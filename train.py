import torch

from utility import compute_gram_matrices, extract_vgg_features


def train_style_transfer_model(content_loader, style_loader, model, num_epochs=10):
    # Optimizer (only train the style conditioner and adapter layers)
    optimizer = torch.optim.Adam([
        {'params': model.style_conditioner.parameters()},
        {'params': model.style_adapter.parameters()}
    ], lr=1e-4)
    
    # Loss functions
    content_criterion = torch.nn.MSELoss()
    style_criterion = torch.nn.MSELoss()  # You could replace with a more sophisticated style loss
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Create paired batches from content and style loaders
        for (content_batch, _), (style_batch, _, _) in zip(content_loader, style_loader):
            content_batch = content_batch.to(model.device)
            style_batch = style_batch.to(model.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Generate stylized images
            stylized_images = model.generate_stylized_image(
                content_batch, 
                style_batch,
                num_inference_steps=10,  # Use fewer steps during training for speed
                guidance_scale=5.0,
                content_strength=0.7
            )
            
            # Compute losses
            # Content loss (perceptual features from VGG)
            content_features = extract_vgg_features(content_batch)
            stylized_features = extract_vgg_features(stylized_images)
            content_loss = content_criterion(content_features, stylized_features)
            
            # Style loss (Gram matrices from VGG)
            style_gram_matrices = compute_gram_matrices(extract_vgg_features(style_batch))
            stylized_gram_matrices = compute_gram_matrices(extract_vgg_features(stylized_images))
            style_loss = style_criterion(style_gram_matrices, stylized_gram_matrices)
            
            # Total loss
            loss = content_loss + style_loss
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'style_conditioner': model.style_conditioner.state_dict(),
                'style_adapter': model.style_adapter.state_dict()
            }, f"style_transfer_checkpoint_epoch{epoch+1}.pth")