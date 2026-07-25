"""
Adaptive Fusion Module for RGB-Thermal Pedestrian Detection
Stage 3: Novel Fusion Strategy Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation)
    Highlights important channels from each modality
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Highlights important regions in each modality
    """
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class AdaptiveFusionModule(nn.Module):
    """
    Complete Fusion Module combining:
    1. Channel Attention for each modality
    2. Spatial Attention for each modality
    3. Learnable weighted fusion
    4. Feature refinement
    """
    
    def __init__(self, in_channels=256, reduction=16, use_spatial=True):
        super().__init__()
        
        self.use_spatial = use_spatial
        
        # Channel attention for each modality
        self.rgb_channel_att = ChannelAttention(in_channels, reduction)
        self.thermal_channel_att = ChannelAttention(in_channels, reduction)
        
        # Spatial attention
        if use_spatial:
            self.rgb_spatial_att = SpatialAttention()
            self.thermal_spatial_att = SpatialAttention()
        
        # Learnable fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_features, thermal_features):
        """
        Args:
            rgb_features: [B, C, H, W]
            thermal_features: [B, C, H, W]
        Returns:
            fused_features: [B, C, H, W]
        """
        # Apply channel attention
        rgb_attended = self.rgb_channel_att(rgb_features)
        thermal_attended = self.thermal_channel_att(thermal_features)
        
        # Apply spatial attention if enabled
        if self.use_spatial:
            rgb_attended = self.rgb_spatial_att(rgb_attended)
            thermal_attended = self.thermal_spatial_att(thermal_attended)
        
        # Adaptive weighted fusion
        weight = torch.sigmoid(self.fusion_weight)
        fused_weighted = weight * rgb_attended + (1 - weight) * thermal_attended
        
        # Concatenate and refine
        concat = torch.cat([rgb_attended, thermal_attended], dim=1)
        refined = self.refinement(concat)
        
        # Combine both paths
        output = fused_weighted + refined
        output = self.output_proj(output)
        
        return output
    
    def get_fusion_weight(self):
        """Get current fusion weight"""
        return torch.sigmoid(self.fusion_weight).item()


class MultiScaleFusion(nn.Module):
    """
    Multi-scale Fusion Module
    Fuses features at different scales for better small object detection
    """
    
    def __init__(self, channels_list=[256, 512, 1024], reduction=16):
        super().__init__()
        
        self.fusion_modules = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        
        for i, channels in enumerate(channels_list):
            self.fusion_modules.append(
                AdaptiveFusionModule(channels, reduction)
            )
            
            # Upsampling for multi-scale fusion
            if i < len(channels_list) - 1:
                self.upsample_ops.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels_list[i+1], 1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    )
                )
        
        print(f"✅ MultiScaleFusion initialized with {len(channels_list)} scales")
    
    def forward(self, rgb_features_list, thermal_features_list):
        """
        Args:
            rgb_features_list: List of RGB features at different scales
            thermal_features_list: List of Thermal features at different scales
        Returns:
            fused_features: List of fused features
        """
        fused_list = []
        
        # Process each scale
        for i, (rgb_feat, thermal_feat) in enumerate(zip(rgb_features_list, thermal_features_list)):
            fused = self.fusion_modules[i](rgb_feat, thermal_feat)
            fused_list.append(fused)
        
        # Multi-scale feature aggregation
        for i in range(len(fused_list) - 2, -1, -1):
            upsampled = F.interpolate(
                fused_list[i+1],
                size=fused_list[i].shape[2:],
                mode='bilinear',
                align_corners=True
            )
            fused_list[i] = fused_list[i] + upsampled
        
        return fused_list


# ====================
# DEMONSTRATION
# ====================

def demo_fusion():
    """Demonstrate the fusion module"""
    
    print("="*70)
    print("🔬 ADAPTIVE FUSION MODULE DEMONSTRATION")
    print("="*70)
    
    # Create sample features
    batch_size = 2
    channels = 256
    height, width = 64, 64
    
    print(f"\n📊 Input Features:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Channels: {channels}")
    print(f"  Resolution: {height}x{width}")
    
    rgb = torch.randn(batch_size, channels, height, width)
    thermal = torch.randn(batch_size, channels, height, width)
    
    print(f"\n  RGB Features Shape: {rgb.shape}")
    print(f"  Thermal Features Shape: {thermal.shape}")
    
    # Create fusion module
    print("\n🔧 Initializing Fusion Module...")
    fusion = AdaptiveFusionModule(channels)
    
    # Forward pass
    print("\n🚀 Forward Pass...")
    fused = fusion(rgb, thermal)
    
    print(f"\n✅ Output Shape: {fused.shape}")
    print(f"✅ Fusion Weight: {fusion.get_fusion_weight():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    # Visualize features
    print("\n📊 Visualizing Features...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Show first sample
    sample_idx = 0
    
    # RGB features (first channel)
    im = axes[0, 0].imshow(rgb[sample_idx, 0].detach().numpy(), cmap='Reds')
    axes[0, 0].set_title('RGB Features (Channel 0)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # RGB features (last channel)
    im = axes[0, 1].imshow(rgb[sample_idx, -1].detach().numpy(), cmap='Reds')
    axes[0, 1].set_title('RGB Features (Last Channel)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Thermal features
    im = axes[0, 2].imshow(thermal[sample_idx, 0].detach().numpy(), cmap='hot')
    axes[0, 2].set_title('Thermal Features (Channel 0)')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Fused features (first channel)
    im = axes[1, 0].imshow(fused[sample_idx, 0].detach().numpy(), cmap='viridis')
    axes[1, 0].set_title('Fused Features (Channel 0)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Fused features (middle channel)
    im = axes[1, 1].imshow(fused[sample_idx, channels//2].detach().numpy(), cmap='viridis')
    axes[1, 1].set_title('Fused Features (Middle Channel)')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Fused features (last channel)
    im = axes[1, 2].imshow(fused[sample_idx, -1].detach().numpy(), cmap='viridis')
    axes[1, 2].set_title('Fused Features (Last Channel)')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('fusion_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to fusion_visualization.png")
    plt.show()
    
    # Feature statistics
    print("\n📊 Feature Statistics:")
    print(f"  RGB - Mean: {rgb.mean():.4f}, Std: {rgb.std():.4f}")
    print(f"  Thermal - Mean: {thermal.mean():.4f}, Std: {thermal.std():.4f}")
    print(f"  Fused - Mean: {fused.mean():.4f}, Std: {fused.std():.4f}")
    
    print("\n" + "="*70)
    print("✅ Fusion Module Ready for Integration!")
    print("="*70)
    
    print("\n📝 How to Integrate:")
    print("  1. Add this module to QFDet neck")
    print("  2. Replace existing fusion with AdaptiveFusionModule")
    print("  3. Fine-tune the model with your dataset")
    print("  4. Evaluate performance improvement")


if __name__ == '__main__':
    demo_fusion()