import os
import argparse
import torch
from torch.utils.data import DataLoader
import lpips

from dataset import ContentDataset, StyleDataset, transform
from style_transfer_diffusion import StyleTransferDiffusion
from train import train_style_transfer_model
from utility import evaluate_model
from memory_utils import free_memory, print_gpu_memory_usage
import config

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU info if available
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print_gpu_memory_usage()
    
    # Initialize datasets
    print(f"Loading content dataset from {args.content_dir}")
    content_dataset = ContentDataset(root_dir=args.content_dir, transform=transform)
    
    print(f"Loading style dataset from {args.style_dir}")
    style_dataset = StyleDataset(root_dir=args.style_dir, transform=transform)
    
    print(f"Content dataset size: {len(content_dataset)}")
    print(f"Style dataset size: {len(style_dataset)}")
    
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
    train_content_loader = DataLoader(content_train_dataset, batch_size=args.batch_size, 
                                      shuffle=True, num_workers=args.num_workers)
    test_content_loader = DataLoader(content_test_dataset, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=args.num_workers)
    train_style_loader = DataLoader(style_train_dataset, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=args.num_workers)
    test_style_loader = DataLoader(style_test_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=args.num_workers)
    
    # Free up memory before loading model
    free_memory()
    
    # Initialize model
    print(f"Initializing model with diffusion model: {args.model_id}")
    model = StyleTransferDiffusion(model_id=args.model_id, device=device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.style_conditioner.load_state_dict(checkpoint['style_conditioner'])
            model.style_adapter.load_state_dict(checkpoint['style_adapter'])
            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {args.checkpoint}")
    
    # Print memory usage after model loading
    print_gpu_memory_usage()
    
    # Train model
    if not args.eval_only:
        print(f"Starting training for {args.epochs} epochs")
        train_style_transfer_model(
            train_content_loader, 
            train_style_loader,
            model,
            num_epochs=args.epochs
        )
        
        # Save final model
        final_model_path = os.path.join(config.CHECKPOINT_DIR, "style_transfer_final_model.pth")
        torch.save({
            'style_conditioner': model.style_conditioner.state_dict(),
            'style_adapter': model.style_adapter.state_dict()
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    # Evaluate model
    if args.evaluate or args.eval_only:
        print("Evaluating model on test set...")
        # Initialize LPIPS model for evaluation
        lpips_model = lpips.LPIPS(net='alex').to(device)
        
        metrics = evaluate_model(model, test_content_loader, test_style_loader, lpips_model)
        print(f"Evaluation Results:")
        print(f"Content Loss: {metrics['content_loss']:.4f}")
        print(f"Style Loss: {metrics['style_loss']:.4f}")
        print(f"LPIPS: {metrics['lpips']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FFT Style Transfer Diffusion Model")
    parser.add_argument("--content-dir", type=str, default=config.CONTENT_IMAGES_DIR,
                        help="Directory containing content images")
    parser.add_argument("--style-dir", type=str, default=config.STYLE_IMAGES_DIR,
                        help="Directory containing style images")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--model-id", type=str, default=config.DIFFUSION_MODEL_ID,
                        help="Hugging Face model ID for the diffusion model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model after training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate model, don't train")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    main(args)