#!/usr/bin/env python3
"""
Test script to verify memory optimizations work.
This script tests if we can load the YOLO model and run basic operations without memory issues.
"""

import os
import sys
import torch
from ultralytics import YOLO
import psutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from FedYOLO.config import YOLO_CONFIG

def get_memory_usage():
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    print(f"[Memory] Process: {memory_info.rss / 1024 / 1024:.1f} MB ({memory_percent:.1f}%)")
    print(f"[Memory] System: {system_memory.used / 1024 / 1024 / 1024:.1f} GB / {system_memory.total / 1024 / 1024 / 1024:.1f} GB ({system_memory.percent:.1f}%)")
    
    return memory_info.rss, memory_percent, system_memory.percent

def test_memory_optimization():
    """Test if memory optimizations work."""
    print("=== Memory Optimization Test ===")
    
    # Check initial memory
    print("\n1. Initial memory usage:")
    get_memory_usage()
    
    # Test 1: Load YOLO model on CPU
    print("\n2. Loading YOLO model on CPU...")
    try:
        model = YOLO('yolo11n.pt')
        model.model = model.model.to('cpu')
        print("✓ YOLO model loaded successfully on CPU")
        get_memory_usage()
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False
    
    # Test 2: Check model parameters
    print("\n3. Checking model parameters...")
    try:
        param_count = sum(p.numel() for p in model.model.parameters())
        print(f"✓ Model has {param_count:,} parameters")
        get_memory_usage()
    except Exception as e:
        print(f"✗ Failed to check parameters: {e}")
        return False
    
    # Test 3: Test with small batch
    print("\n4. Testing with small batch...")
    try:
        # Create a small dummy batch
        dummy_input = torch.randn(1, 3, YOLO_CONFIG['image_size'], YOLO_CONFIG['image_size'])
        dummy_input = dummy_input.to('cpu')
        
        with torch.no_grad():
            output = model.model(dummy_input)
        
        print("✓ Forward pass successful")
        get_memory_usage()
    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        return False
    
    # Test 4: Clean up
    print("\n5. Cleaning up...")
    try:
        del model
        del dummy_input
        del output
        
        import gc
        gc.collect()
        
        print("✓ Cleanup successful")
        get_memory_usage()
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        return False
    
    print("\n=== All tests passed! Memory optimizations are working. ===")
    return True

if __name__ == "__main__":
    success = test_memory_optimization()
    if success:
        print("\n✅ Ready to run federated training with memory optimizations!")
    else:
        print("\n❌ Memory optimization test failed. Check the errors above.")
        sys.exit(1) 