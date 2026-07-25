"""
Create mock weights for testing Stage 2 evaluation
Run: python create_mock_weights.py
"""

import torch
import os

def create_mock_weights():
    """Create mock model weights for testing"""
    
    # Create weights directory
    os.makedirs('weights', exist_ok=True)
    
    print("Creating mock weights for testing...")
    
    # Create a proper mock model state dict with realistic shapes
    mock_state = {
        'state_dict': {
            # Backbone (ResNet50)
            'backbone.conv1.weight': torch.randn(64, 3, 7, 7),
            'backbone.bn1.weight': torch.ones(64),
            'backbone.bn1.bias': torch.zeros(64),
            'backbone.bn1.running_mean': torch.zeros(64),
            'backbone.bn1.running_var': torch.ones(64),
            'backbone.layer1.0.conv1.weight': torch.randn(64, 64, 1, 1),
            'backbone.layer1.0.bn1.weight': torch.ones(64),
            'backbone.layer1.0.bn1.bias': torch.zeros(64),
            'backbone.layer2.0.conv1.weight': torch.randn(128, 256, 1, 1),
            'backbone.layer2.0.bn1.weight': torch.ones(128),
            'backbone.layer2.0.bn1.bias': torch.zeros(128),
            'backbone.layer3.0.conv1.weight': torch.randn(256, 512, 1, 1),
            'backbone.layer3.0.bn1.weight': torch.ones(256),
            'backbone.layer3.0.bn1.bias': torch.zeros(256),
            'backbone.layer4.0.conv1.weight': torch.randn(512, 1024, 1, 1),
            'backbone.layer4.0.bn1.weight': torch.ones(512),
            'backbone.layer4.0.bn1.bias': torch.zeros(512),
            
            # Neck (FPN)
            'neck.fpn_convs.0.weight': torch.randn(256, 256, 3, 3),
            'neck.fpn_convs.0.bias': torch.zeros(256),
            'neck.fpn_convs.1.weight': torch.randn(256, 256, 3, 3),
            'neck.fpn_convs.1.bias': torch.zeros(256),
            'neck.fpn_convs.2.weight': torch.randn(256, 256, 3, 3),
            'neck.fpn_convs.2.bias': torch.zeros(256),
            'neck.fpn_convs.3.weight': torch.randn(256, 256, 3, 3),
            'neck.fpn_convs.3.bias': torch.zeros(256),
            
            # RPN Head
            'rpn_head.rpn_conv.weight': torch.randn(256, 256, 3, 3),
            'rpn_head.rpn_conv.bias': torch.zeros(256),
            'rpn_head.rpn_cls.weight': torch.randn(3, 256, 1, 1),
            'rpn_head.rpn_cls.bias': torch.zeros(3),
            'rpn_head.rpn_reg.weight': torch.randn(12, 256, 1, 1),
            'rpn_head.rpn_reg.bias': torch.zeros(12),
            
            # RoI Head
            'roi_head.bbox_roi_extractor.featmap_strides': torch.tensor([4, 8, 16, 32]),
            'roi_head.bbox_head.fc_cls.weight': torch.randn(2, 1024),
            'roi_head.bbox_head.fc_cls.bias': torch.zeros(2),
            'roi_head.bbox_head.fc_reg.weight': torch.randn(8, 1024),
            'roi_head.bbox_head.fc_reg.bias': torch.zeros(8),
        }
    }
    
    # Save mock weights for all three models
    weights_files = ['qfdet_weights.pth', 'rgb_weights.pth', 'thermal_weights.pth']
    
    for weight_file in weights_files:
        filepath = os.path.join('weights', weight_file)
        torch.save(mock_state, filepath)
        print(f'✅ Created: {filepath}')
    
    # Show file sizes
    print('\n✅ All mock weights created successfully!')
    print('📂 Location: weights/')
    print('\n📊 File sizes:')
    
    for weight_file in weights_files:
        filepath = os.path.join('weights', weight_file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f'  - {weight_file}: {size:.2f} MB')
    
    return True

if __name__ == '__main__':
    create_mock_weights()
    print('\n✅ You can now run the evaluation script:')
    print('   python tools/evaluate_stage2.py --device cpu --model all')