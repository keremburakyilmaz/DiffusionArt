import torch
from torch.utils.data import DataLoader
from dataset import ContentDataset, StyleDataset, transform
from style_transfer_diffusion import StyleTransferDiffusion
from train import train_style_transfer_model
from utility import evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datasets and dataloaders
    content_dataset = ContentDataset(root_dir="raw_images\content", transform=transform)
    style_dataset = StyleDataset(root_dir="raw_images\style", transform=transform)
    
    # Split datasets
    content_train_size = int(0.8 * len(content_dataset))
    content_test_size = len(content_dataset) - content_train_size
    style_train_size = int(0.8 * len(style_dataset))
    style_test_size = len(style_dataset) - style_train_size
    
    content_train_dataset, content_test_dataset = torch.utils.data.random_split(
        content_dataset, [content_train_size, content_test_size])
    style_train_dataset, style_test_dataset = torch.utils.data.random_split(
        style_dataset, [style_train_size, style_test_size])
    
    # Create dataloaders
    train_content_loader = DataLoader(content_train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_content_loader = DataLoader(content_test_dataset, batch_size=4, shuffle=False, num_workers=4)
    train_style_loader = DataLoader(style_train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_style_loader = DataLoader(style_test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Initialize model
    model = StyleTransferDiffusion(model_id="CompVis/stable-diffusion-v1-4", device=device)
    
    # Initialize LPIPS model for evaluation
    import lpips
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Train model
    train_style_transfer_model(train_content_loader, train_style_loader, model, num_epochs=10)
    
    # Evaluate model
    metrics = evaluate_model(model, test_content_loader, test_style_loader, lpips_model)
    print(f"Evaluation Results:")
    print(f"Content Loss: {metrics['content_loss']}")
    print(f"Style Loss: {metrics['style_loss']}")
    print(f"LPIPS: {metrics['lpips']}")
    
    # Save final model
    torch.save({
        'style_conditioner': model.style_conditioner.state_dict(),
        'style_adapter': model.style_adapter.state_dict()
    }, "style_transfer_final_model.pth")

if __name__ == "__main__":
    main()