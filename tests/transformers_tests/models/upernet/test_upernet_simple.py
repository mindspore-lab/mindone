#!/usr/bin/env python
"""Simple test script for UperNet MindSpore implementation"""

import sys
sys.path.insert(0, '/mnt/e/github/convert/mindone')

import numpy as np
import mindspore as ms
from mindspore import Tensor


def test_upernet_mindspore():
    """Test UperNet MindSpore implementation independently"""
    print("Testing UperNet MindSpore implementation...")
    
    try:
        # Import UperNet components
        from mindone.transformers.models.upernet import UperNetForSemanticSegmentation, UperNetConfig
        from mindone.transformers.models.convnext import ConvNextConfig
        print("✓ UperNet imports successful")
        
        # Create ConvNext backbone config
        backbone_config = ConvNextConfig(
            num_channels=3,
            hidden_sizes=[16, 32, 64, 128],
            depths=[1, 1, 1, 1],
            out_features=['stage1', 'stage2', 'stage3', 'stage4'],
        )
        print("✓ ConvNextConfig creation successful")
        
        # Create UperNet config
        config = UperNetConfig(
            backbone_config=backbone_config,
            hidden_size=32,
            num_labels=10,
            pool_scales=[1, 2, 3, 6],
            use_auxiliary_head=False,  # Disable auxiliary head for simplicity
        )
        print("✓ UperNetConfig creation successful")
        
        # Create model
        model = UperNetForSemanticSegmentation(config)
        print("✓ UperNet model creation successful")
        
        # Test forward pass
        batch_size = 2
        height, width = 32, 32
        pixel_values = Tensor(np.random.randn(batch_size, 3, height, width).astype(np.float32))
        
        print(f"Testing forward pass with input shape: {pixel_values.shape}")
        
        # Set model to eval mode
        model.set_train(False)
        
        # Forward pass
        outputs = model(pixel_values)
        
        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  Expected shape: ({batch_size}, {config.num_labels}, {height}, {width})")
        
        # Verify output shape
        expected_shape = (batch_size, config.num_labels, height, width)
        assert outputs.logits.shape == expected_shape, f"Expected shape {expected_shape}, got {outputs.logits.shape}"
        
        print("✓ Output shape validation successful")
        
        # Test with labels
        labels = Tensor(np.random.randint(0, config.num_labels, size=(batch_size, height, width)).astype(np.int32))
        outputs_with_loss = model(pixel_values, labels=labels)
        
        print(f"✓ Forward pass with labels successful")
        print(f"  Loss: {outputs_with_loss.loss}")
        
        print("\n🎉 All tests passed! UperNet MindSpore implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_upernet_mindspore()
    sys.exit(0 if success else 1)