import torch


class FFTStyleConditioner(torch.nn.Module):
    def __init__(self, low_freq_percent=0.1, mid_freq_percent=0.4):
        super(FFTStyleConditioner, self).__init__()
        self.low_freq_percent = low_freq_percent
        self.mid_freq_percent = mid_freq_percent
        
        # Convolutional layers to process different frequency bands
        self.low_freq_processor = torch.nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.mid_freq_processor = torch.nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.high_freq_processor = torch.nn.Conv2d(6, 64, kernel_size=3, padding=1)
        
        # Final projection layer
        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(64*3, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        )
        
    def extract_frequency_bands(self, image):
        # Apply FFT
        fft = torch.fft.fft2(image, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # Get dimensions
        B, C, H, W = fft_shifted.shape
        center_h, center_w = H // 2, W // 2
        
        # Create masks for different frequency bands
        mask_low = torch.zeros((H, W), device=image.device)
        mask_mid = torch.zeros((H, W), device=image.device)
        mask_high = torch.ones((H, W), device=image.device)
        
        # Calculate radii for frequency bands
        r_low = int(min(H, W) * self.low_freq_percent)
        r_mid = int(min(H, W) * (self.low_freq_percent + self.mid_freq_percent))
        
        # Create circle masks
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=image.device), 
                                        torch.arange(W, device=image.device))
        dist_from_center = ((y_grid - center_h)**2 + (x_grid - center_w)**2) ** 0.5
        
        # Apply masks
        mask_low[dist_from_center <= r_low] = 1.0
        mask_mid[(dist_from_center > r_low) & (dist_from_center <= r_mid)] = 1.0
        mask_high[dist_from_center <= r_mid] = 0.0
        
        # Expand masks for broadcasting
        mask_low = mask_low.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        mask_mid = mask_mid.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        
        # Apply masks to FFT result
        fft_low = fft_shifted * mask_low
        fft_mid = fft_shifted * mask_mid
        fft_high = fft_shifted * mask_high
        
        # Inverse FFT to get filtered images
        low_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_low, dim=(-2, -1)), dim=(-2, -1)).real
        mid_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_mid, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)), dim=(-2, -1)).real
        
        return low_freq, mid_freq, high_freq
    
    def forward(self, content_img, style_img):
        # Extract frequency bands for content and style
        content_low, content_mid, content_high = self.extract_frequency_bands(content_img)
        style_low, style_mid, style_high = self.extract_frequency_bands(style_img)
        
        # Process each frequency band (concatenate content and style for each band)
        low_concat = torch.cat([content_low, style_low], dim=1)
        mid_concat = torch.cat([content_mid, style_mid], dim=1)
        high_concat = torch.cat([content_high, style_high], dim=1)
        
        # Process each frequency band
        low_features = self.low_freq_processor(low_concat)
        mid_features = self.mid_freq_processor(mid_concat)
        high_features = self.high_freq_processor(high_concat)
        
        # Combine all features
        combined_features = torch.cat([low_features, mid_features, high_features], dim=1)
        style_condition = self.projection(combined_features)
        
        return style_condition