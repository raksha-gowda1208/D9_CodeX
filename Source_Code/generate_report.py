"""
Generate Stage 2 Report
Run: python generate_report.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_results():
    """Load all evaluation results"""
    results_dir = 'outputs'
    results = {}
    
    for model_file in ['rgb_results.json', 'thermal_results.json', 'baseline_results.json']:
        filepath = os.path.join(results_dir, model_file)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                model_name = model_file.replace('_results.json', '').capitalize()
                results[model_name] = json.load(f)
                print(f"✅ Loaded: {model_name}")
    
    return results

def create_comparison_table(results):
    """Create comparison DataFrame"""
    data = []
    for model_name, result in results.items():
        det = result['detection']
        comp = result['computational']
        data.append({
            'Model': model_name,
            'mAP': det['mAP'],
            'AP50': det['AP50'],
            'AP75': det['AP75'],
            'AP_s': det['AP_s'],
            'AP_m': det['AP_m'],
            'AP_l': det['AP_l'],
            'AR_100': det['AR_100'],
            'FPS': comp['fps'],
            'Params (M)': comp['total_params'] / 1e6,
            'Size (MB)': comp['model_size_mb']
        })
    
    return pd.DataFrame(data)

def print_tables(df):
    """Print formatted tables"""
    print("\n" + "="*80)
    print("📊 STAGE 2: UNIMODAL ANALYSIS RESULTS")
    print("="*80)
    
    print("\n🎯 Detection Performance:")
    det_cols = ['Model', 'mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l', 'AR_100']
    print(tabulate(df[det_cols], headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    print("\n⚡ Computational Performance:")
    comp_cols = ['Model', 'FPS', 'Params (M)', 'Size (MB)']
    print(tabulate(df[comp_cols], headers='keys', tablefmt='grid', floatfmt='.2f'))
    
    # Save to CSV
    df.to_csv('stage2_comparison.csv', index=False)
    print("\n✅ Comparison saved to stage2_comparison.csv")

def create_visualizations(df):
    """Create visualization plots"""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Detection Metrics Bar Chart
    metrics = ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, row in df.iterrows():
        values = [row[m] for m in metrics]
        ax1.bar(x + i*width, values, width, label=row['Model'])
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Detection Performance Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # 2. Small Object Detection Focus
    small_metrics = ['AP_s', 'AP_m', 'AP_l']
    x = np.arange(len(small_metrics))
    
    for i, row in df.iterrows():
        values = [row[m] for m in small_metrics]
        ax2.bar(x + i*width, values, width, label=row['Model'])
    
    ax2.set_xlabel('Object Size')
    ax2.set_ylabel('AP Score')
    ax2.set_title('Performance by Object Size')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(['Small', 'Medium', 'Large'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # 3. Computational Metrics
    comp_metrics = ['FPS', 'Params (M)', 'Size (MB)']
    x = np.arange(len(comp_metrics))
    
    # Normalize for visualization
    normalized_data = []
    for row in df.iterrows():
        idx, data = row
        normalized = [
            data['FPS'] / df['FPS'].max(),
            data['Params (M)'] / df['Params (M)'].max(),
            data['Size (MB)'] / df['Size (MB)'].max()
        ]
        normalized_data.append(normalized)
    
    for i, row in enumerate(df.iterrows()):
        idx, data = row
        ax3.bar(x + i*width, normalized_data[i], width, label=data['Model'])
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Normalized Score')
    ax3.set_title('Computational Performance (Normalized)')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['FPS', 'Params', 'Size'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Radar Chart
    from math import pi
    categories = ['mAP', 'AP_s', 'AP_m', 'AP_l', 'FPS']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for _, row in df.iterrows():
        values = [
            row['mAP'],
            row['AP_s'],
            row['AP_m'],
            row['AP_l'],
            row['FPS'] / 20  # Normalize FPS to ~0-1 range
        ]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax4.fill(angles, values, alpha=0.1)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Model Performance Radar')
    ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('stage2_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to stage2_comparison.png")
    plt.show()

def analyze_findings(df):
    """Analyze key findings"""
    print("\n" + "="*80)
    print("📈 KEY FINDINGS & ANALYSIS")
    print("="*80)
    
    # Find best models
    best_map = df.loc[df['mAP'].idxmax()]
    best_fps = df.loc[df['FPS'].idxmax()]
    best_small = df.loc[df['AP_s'].idxmax()]
    
    print("\n🏆 Best Performers:")
    print(f"  • Best Overall (mAP): {best_map['Model']} ({best_map['mAP']:.4f})")
    print(f"  • Best Speed (FPS): {best_fps['Model']} ({best_fps['FPS']:.2f})")
    print(f"  • Best Small Object: {best_small['Model']} ({best_small['AP_s']:.4f})")
    
    # Compare modalities
    print("\n🔍 Modality Strengths:")
    
    rgb_row = df[df['Model'] == 'RGB_Only']
    thermal_row = df[df['Model'] == 'Thermal_Only']
    baseline_row = df[df['Model'] == 'QFDet_Baseline']
    
    if not rgb_row.empty and not thermal_row.empty:
        rgb = rgb_row.iloc[0]
        thermal = thermal_row.iloc[0]
        
        if rgb['AP_s'] > thermal['AP_s']:
            print(f"  • RGB excels at small objects (+{rgb['AP_s'] - thermal['AP_s']:.3f})")
        else:
            print(f"  • Thermal excels at small objects (+{thermal['AP_s'] - rgb['AP_s']:.3f})")
        
        if rgb['mAP'] > thermal['mAP']:
            print(f"  • RGB has better overall accuracy (+{rgb['mAP'] - thermal['mAP']:.3f})")
        else:
            print(f"  • Thermal has better overall accuracy (+{thermal['mAP'] - rgb['mAP']:.3f})")
    
    # Baseline improvement
    if not baseline_row.empty and not rgb_row.empty and not thermal_row.empty:
        baseline = baseline_row.iloc[0]
        best_modality = max(rgb_row.iloc[0]['mAP'], thermal_row.iloc[0]['mAP'])
        improvement = baseline['mAP'] - best_modality
        
        if improvement > 0:
            print(f"\n✅ Fusion Improvement: +{improvement:.3f} mAP over best unimodal")
        else:
            print(f"\n⚠️ Fusion needs improvement: -{abs(improvement):.3f} mAP vs best unimodal")
    
    print("\n💡 Recommendations for Stage 3:")
    print("  1. Focus on improving small object detection (AP_s)")
    print("  2. Consider adaptive fusion based on lighting conditions")
    print("  3. Explore attention mechanisms for better feature fusion")

def main():
    print("Generating Stage 2 Report...")
    print("-" * 50)
    
    # Load results
    results = load_results()
    
    if not results:
        print("❌ No results found. Please run evaluation first:")
        print("   python tools/evaluate_stage2.py --device cpu --model all")
        return
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Print tables
    print_tables(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Analyze findings
    analyze_findings(df)
    
    print("\n" + "="*80)
    print("✅ REPORT GENERATION COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("  • stage2_comparison.csv - Data table")
    print("  • stage2_comparison.png - Visualization plots")
    print("\n📋 Next Steps:")
    print("  1. Review the results and identify strengths/weaknesses")
    print("  2. Proceed to Stage 3: Develop fusion strategy")
    print("  3. Document findings in your technical report")

if __name__ == '__main__':
    main()