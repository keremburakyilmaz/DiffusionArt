import torch
import torchvision


def extract_vgg_features(images):
    """Extract features from VGG network for perceptual loss"""
    # Use a pretrained VGG model
    vgg = torchvision.models.vgg16(pretrained=True).features.eval().to(images.device)
    
    # Don't compute gradients for VGG
    for param in vgg.parameters():
        param.requires_grad = False
    
    # Extract features from specific layers
    features = []
    x = images
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in [3, 8, 15, 22]:  # conv1_2, conv2_2, conv3_3, conv4_3
            features.append(x)
    
    return features

def compute_gram_matrices(features):
    """Compute Gram matrices for style loss"""
    gram_matrices = []
    
    for feature in features:
        B, C, H, W = feature.size()
        feature = feature.view(B, C, H * W)
        gram = torch.bmm(feature, feature.transpose(1, 2))
        # Normalize
        gram = gram.div(C * H * W)
        gram_matrices.append(gram)
    
    return gram_matrices

def calculate_lpips(image1, image2, lpips_model):
    """Calculate LPIPS perceptual similarity"""
    return lpips_model(image1, image2).mean()

def evaluate_model(model, test_content_loader, test_style_loader, lpips_model):
    """Evaluate model performance"""
    content_losses = []
    style_losses = []
    lpips_values = []
    
    with torch.no_grad():
        for (content_batch, _), (style_batch, _, _) in zip(test_content_loader, test_style_loader):
            content_batch = content_batch.to(model.device)
            style_batch = style_batch.to(model.device)
            
            # Generate stylized images
            stylized_images = model.generate_stylized_image(
                content_batch, 
                style_batch,
                num_inference_steps=50,
                guidance_scale=7.5,
                content_strength=0.8
            )
            
            # Compute metrics
            content_features = extract_vgg_features(content_batch)
            style_features = extract_vgg_features(style_batch)
            stylized_features = extract_vgg_features(stylized_images)
            
            # Content loss
            content_loss = torch.nn.MSELoss()(content_features[-1], stylized_features[-1])
            content_losses.append(content_loss.item())
            
            # Style loss (using Gram matrices)
            style_gram = compute_gram_matrices(style_features)
            stylized_gram = compute_gram_matrices(stylized_features)
            style_loss = torch.nn.MSELoss()(style_gram[0], stylized_gram[0])
            style_losses.append(style_loss.item())
            
            # LPIPS
            lpips_value = calculate_lpips(content_batch, stylized_images, lpips_model)
            lpips_values.append(lpips_value.item())
    
    return {
        'content_loss': sum(content_losses) / len(content_losses),
        'style_loss': sum(style_losses) / len(style_losses),
        'lpips': sum(lpips_values) / len(lpips_values)
    }