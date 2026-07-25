"""
Create sample results for testing
Run: python create_sample_results.py
"""

import json
import os

def create_sample_results():
    """Create sample result files"""
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Define sample results
    sample_results = {
        'model': 'RGB_Only',
        'detection': {
            'mAP': 0.4523,
            'AP50': 0.6891,
            'AP75': 0.4912,
            'AP_s': 0.2134,
            'AP_m': 0.4521,
            'AP_l': 0.6123,
            'AR_1': 0.3512,
            'AR_10': 0.5123,
            'AR_100': 0.6234
        },
        'computational': {
            'total_params': 42567890,
            'trainable_params': 42567890,
            'model_size_mb': 162.4,
            'fps': 18.5,
            'avg_inference_ms': 54.1
        },
        'inference_time': 45.6
    }
    
    # Create variations for different models
    models = {
        'rgb_results.json': ('RGB_Only', 0.4523, 0.2134, 18.5),
        'thermal_results.json': ('Thermal_Only', 0.4231, 0.2856, 18.2),
        'baseline_results.json': ('QFDet_Baseline', 0.4867, 0.3245, 16.8)
    }
    
    for filename, (name, mAP, ap_s, fps) in models.items():
        # Create a copy with modified values
        result = {
            'model': name,
            'detection': {
                'mAP': mAP,
                'AP50': mAP * 1.5,  # AP50 is usually higher
                'AP75': mAP * 0.9,  # AP75 is usually lower
                'AP_s': ap_s,
                'AP_m': mAP * 0.95,
                'AP_l': mAP * 1.1,
                'AR_1': 0.35,
                'AR_10': 0.51,
                'AR_100': 0.62
            },
            'computational': {
                'total_params': 42567890,
                'trainable_params': 42567890,
                'model_size_mb': 162.4,
                'fps': fps,
                'avg_inference_ms': 1000 / fps if fps > 0 else 0
            },
            'inference_time': 45.6
        }
        
        filepath = os.path.join('outputs', filename)
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
            print(f"✅ Created: {filepath}")

if __name__ == '__main__':
    create_sample_results()
    print("\n✅ Sample results created!")
    print("\nNow run: python generate_report.py")