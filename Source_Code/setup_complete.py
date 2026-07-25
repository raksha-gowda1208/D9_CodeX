"""
Complete setup script for Stage 2 Hackathon
Run: python setup_complete.py
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and print output"""
    print(f"\n▶ Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error occurred! Return code: {result.returncode}")
        print(f"Error output: {result.stderr}")
        return False
    print(result.stdout)
    return True

def main():
    print("="*60)
    print("  Complete Setup for Stage 2 Hackathon")
    print("="*60)
    
    # 1. Upgrade pip and install setuptools
    print("\n[1/7] Upgrading pip and installing setuptools...")
    run_command("python -m pip install --upgrade pip setuptools wheel")
    
    # 2. Install core dependencies
    print("\n[2/7] Installing core dependencies...")
    deps = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "tqdm>=4.62.0",
        "tabulate>=0.8.9",
        "pycocotools>=2.0.2",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "terminaltables>=3.1.0"
    ]
    for dep in deps:
        run_command(f"pip install {dep}")
    
    # 3. Install PyTorch
    print("\n[3/7] Installing PyTorch...")
    run_command("pip install torch torchvision")
    
    # 4. Install mmcv
    print("\n[4/7] Installing mmcv...")
    run_command("pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html")
    
    # 5. Install mmdet
    print("\n[5/7] Installing mmdet...")
    run_command("pip install mmdet")
    
    # 6. Check QFDet directory
    print("\n[6/7] Setting up QFDet path...")
    qfdet_dir = os.path.join(os.getcwd(), "mmdet-rgbtdroneperson")
    if os.path.exists(qfdet_dir):
        print(f"✅ Found QFDet at: {qfdet_dir}")
        # Add to Python path
        with open("qfdet_path.pth", "w") as f:
            f.write(qfdet_dir)
        print(f"✅ Added QFDet to Python path")
    else:
        print(f"⚠️ QFDet directory not found at: {qfdet_dir}")
        print("   Please clone it manually:")
        print("   git clone https://github.com/NNNNerd/mmdet-rgbtdroneperson.git")
    
    # 7. Verify installation
    print("\n[7/7] Verifying installation...")
    verify_script = """
import sys
import os

# Add QFDet path if it exists
qfdet_path = os.path.join(os.getcwd(), 'mmdet-rgbtdroneperson')
if os.path.exists(qfdet_path):
    sys.path.insert(0, qfdet_path)

try:
    import mmcv
    print(f"✅ mmcv: {mmcv.__version__}")
except ImportError as e:
    print(f"❌ mmcv: {e}")

try:
    import mmdet
    print(f"✅ mmdet: {mmdet.__version__}")
except ImportError as e:
    print(f"❌ mmdet: {e}")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")

try:
    import cv2
    print(f"✅ opencv: {cv2.__version__}")
except ImportError as e:
    print(f"❌ opencv: {e}")

try:
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")
"""
    run_command(f'python -c "{verify_script}"')
    
    print("\n" + "="*60)
    print("  Setup Complete!")
    print("="*60)
    print("\nTo run evaluation:")
    print("  python tools/evaluate_unimodal.py --device cpu --model all")
    print("\nIf you get import errors, add this to your script:")
    print("  import sys")
    print("  sys.path.insert(0, 'mmdet-rgbtdroneperson')")

if __name__ == "__main__":
    main()