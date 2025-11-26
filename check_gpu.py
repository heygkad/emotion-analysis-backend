#!/usr/bin/env python3
"""
Quick script to verify TensorFlow Metal GPU support on Mac
Run this before running model.py to diagnose any issues
"""

import sys
import os

print("=" * 60)
print("TensorFlow Metal GPU Support Checker")
print("=" * 60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check if TensorFlow is installed
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow is NOT installed!")
    print("  Install with: pip install tensorflow-macos")
    sys.exit(1)

# Check if tensorflow-metal is installed (it's a plugin, doesn't need direct import)
try:
    import tensorflow_metal
    print(f"✓ tensorflow-metal is installed")
    try:
        print(f"  Version: {tensorflow_metal.__version__}")
    except:
        pass
except ImportError:
    # tensorflow-metal is a plugin, doesn't need direct import
    # Check if it's installed via pip
    import subprocess
    result = subprocess.run(['pip', 'show', 'tensorflow-metal'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ tensorflow-metal is installed (plugin loaded automatically)")
    else:
        print("⚠ tensorflow-metal may not be installed")
        print("  Install with: pip install tensorflow-metal")

# Configure environment
os.environ['TF_METAL_DEVICE_PLACEMENT'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# List all devices
print("\nAll available devices:")
all_devices = tf.config.list_physical_devices()
if not all_devices:
    print("  ⚠ No devices found!")
else:
    for device in all_devices:
        print(f"  - {device}")

# Check for GPU
gpu_devices = tf.config.list_physical_devices('GPU')
cpu_devices = tf.config.list_physical_devices('CPU')

print(f"\nGPU devices: {len(gpu_devices)}")
if gpu_devices:
    for gpu in gpu_devices:
        print(f"  ✓ {gpu}")
else:
    print("  ✗ No GPU devices found")

print(f"\nCPU devices: {len(cpu_devices)}")
for cpu in cpu_devices:
    print(f"  - {cpu}")

# Try to create a simple tensor on GPU
print("\n" + "=" * 60)
print("Testing GPU computation...")
print("=" * 60)

if gpu_devices:
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"✓ Successfully created tensor on GPU")
            print(f"  Result: {c.numpy()}")
            print("\n✓ GPU (MPS) is working correctly!")
    except Exception as e:
        print(f"✗ Error using GPU: {e}")
        print("  GPU may not be fully functional")
else:
    print("⚠ No GPU available - cannot test GPU computation")
    print("\nTroubleshooting:")
    print("1. Reinstall: pip uninstall tensorflow-macos tensorflow-metal")
    print("              pip install tensorflow-macos tensorflow-metal")
    print("2. Check macOS version (should be macOS 12.0+)")
    print("3. Verify you're on Apple Silicon (M1/M2/M3/M4)")
    print("4. Try restarting your terminal/Python environment")

print("=" * 60)

