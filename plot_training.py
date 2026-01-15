#!/usr/bin/env python
"""
Detailed training comparison - Solve curve overlapping problem
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_comparison_detailed(csv_paths_without, csv_paths_with, dataset_names, output_dir='comparison_plots'):
    """
    Detailed comparison of training curves with/without concat - various visualization methods
    """
    
    Path(output_dir).mkdir(exist_ok=True)
    
    colors = {
        'without': '#E63946',  # red
        'with': '#06AED5'      # blue
    }
    
    for dataset_idx, dataset_name in enumerate(dataset_names):
        print(f"\n{'='*70}")
        print(f"📊 Processing: {dataset_name}")
        print(f"{'='*70}")
        
        # Load data
        df_without = pd.read_csv(csv_paths_without[dataset_idx])
        df_with = pd.read_csv(csv_paths_with[dataset_idx])
        
        max_steps = min(df_without['step'].max(), df_with['step'].max())
        df_without = df_without[df_without['step'] <= max_steps]
        df_with = df_with[df_with['step'] <= max_steps]
        
        print(f"   Comparing up to step: {max_steps}")
        
        # Create 3 types of visualizations
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        
        # 1. Raw curve (top row, wide)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # 2. Difference curve (middle)
        ax2 = fig.add_subplot(gs[1, 0])
        
        # 3. Sliding window average difference (bottom)
        ax4 = fig.add_subplot(gs[2, 0])
        
        loss_col = 'train/loss_step'
        
        # Get data
        data_without = df_without[['step', loss_col]].dropna()
        data_with = df_with[['step', loss_col]].dropna()
        
        if len(data_without) == 0 or len(data_with) == 0:
            print(f"   ⚠️ No data for {loss_col}")
            continue
        
        # === Fig 1: Raw curve (just with/without, no moving average) ===
        ax1.plot(data_without['step'], data_without[loss_col], 
                label='Without Concat', linewidth=1.5, alpha=0.8, color=colors['without'])
        ax1.plot(data_with['step'], data_with[loss_col], 
                label='With Concat', linewidth=1.5, alpha=0.8, color=colors['with'])
        
        ax1.set_xlabel('Steps', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title(f'{dataset_name} - Training Loss Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # === Fig 2: Difference curve (With - Without) ===
        # Align steps
        common_steps = np.intersect1d(data_without['step'].values, data_with['step'].values)
        
        diff_values = []
        for step in common_steps:
            val_without = data_without[data_without['step'] == step][loss_col].values[0]
            val_with = data_with[data_with['step'] == step][loss_col].values[0]
            diff_values.append(val_with - val_without)
        
        diff_values = np.array(diff_values)
        
        ax2.plot(common_steps, diff_values, linewidth=1.5, alpha=0.7, color='#2A9D8F')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(common_steps, 0, diff_values, 
                        where=(diff_values < 0), alpha=0.3, color='green', label='With < Without (Better)')
        ax2.fill_between(common_steps, 0, diff_values, 
                        where=(diff_values >= 0), alpha=0.3, color='red', label='With > Without (Worse)')
        
        ax2.set_xlabel('Steps', fontsize=10)
        ax2.set_ylabel('Loss Difference (With - Without)', fontsize=10)
        ax2.set_title('Loss Difference Over Training', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # === Fig 3: Sliding window average difference ===
        if len(diff_values) > 100:
            window = min(100, len(diff_values) // 5)
            diff_smoothed = pd.Series(diff_values).rolling(window=window, center=True).mean()
            
            ax4.plot(common_steps, diff_smoothed, linewidth=2.5, color='#E76F51')
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax4.fill_between(common_steps, 0, diff_smoothed, 
                            where=(diff_smoothed < 0), alpha=0.3, color='green')
            ax4.fill_between(common_steps, 0, diff_smoothed, 
                            where=(diff_smoothed >= 0), alpha=0.3, color='red')
            
            ax4.set_xlabel('Steps', fontsize=10)
            ax4.set_ylabel(f'Smoothed Difference (MA-{window})', fontsize=10)
            ax4.set_title('Smoothed Loss Difference', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # Save
        output_path = f'{output_dir}/{dataset_name}_detailed_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()
        
        # Print statistics
        print(f"\n   📊 Statistics:")
        print(f"      Mean difference: {np.mean(diff_values):.6f}")
        print(f"      Std difference: {np.std(diff_values):.6f}")
        print(f"      Final difference: {diff_values[-1]:.6f}")
        # Percentage difference metrics removed

def main():
    csv_paths_without = [
        '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-04T09-59-07_shapes3d-vq-4-16-encdiff23/testtube/version_1/metrics.csv',
        '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-05T06-14-44_mpi3d-vq-4-16-encdiff23/testtube/version_0/metrics.csv',
        '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-05T01-46-21_cars3d-vq-4-16-encdiff23/testtube/version_0/metrics.csv'
    ]
    
    csv_paths_with = [
        '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T07-42-42_shapes3d-vq-4-16-encdiff23/testtube/version_0/metrics.csv',
        '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T08-23-34_mpi3d-vq-4-16-encdiff23/testtube/version_0/metrics.csv',
        '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T08-25-57_cars3d-vq-4-16-encdiff23/testtube/version_0/metrics.csv'
    ]
    
    dataset_names = ['Shapes3D', 'MPI3D', 'Cars3D']
    
    plot_comparison_detailed(csv_paths_without, csv_paths_with, dataset_names)
    
    print("\n" + "=" * 70)
    print("✅ Detailed comparison plots generated!")
    print("=" * 70)

if __name__ == "__main__":
    main()