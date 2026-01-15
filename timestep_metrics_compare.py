#!/usr/bin/env python
"""
Timestep Metrics Compare
========================
This script extracts FactorVAE (eval_accuracy) and DCI (disentanglement) 
metrics from each timestep's JSON file and creates a summary table CSV 
in each experiment directory.
"""

import os
import json
import glob
import pandas as pd

# Experiment directories (w/ and w/o concat cases)
EXPERIMENT_DIRS = [
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-04T09-59-07_shapes3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-05T01-46-21_cars3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-05T06-14-44_mpi3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T07-42-42_shapes3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T08-23-34_mpi3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T08-25-57_cars3d-vq-4-16-encdiff23',
]

METRICS_FOLDER = 'metrics_sin'
OUTPUT_FILENAME = 'timestep_metrics_summary.csv'
TOP_K = 10  # Keep timesteps that are in top K for either FactorVAE or DCI


def extract_metrics_from_json(json_path):
    """
    Extract FactorVAE eval_accuracy and DCI disentanglement from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        dict with 'factor_vae_eval_accuracy' and 'dci_disentanglement' keys,
        or None values if metrics are missing
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        factor_vae_eval = data.get('factor_VAE', {}).get('eval_accuracy', None)
        dci_disentangle = data.get('dci', {}).get('disentanglement', None)
        
        return {
            'factor_vae_eval_accuracy': factor_vae_eval,
            'dci_disentanglement': dci_disentangle
        }
    except Exception as e:
        print(f"  Warning: Error reading {json_path}: {e}")
        return {
            'factor_vae_eval_accuracy': None,
            'dci_disentanglement': None
        }


def process_experiment_dir(exp_dir):
    """
    Process a single experiment directory, extracting metrics from all timestep JSONs.
    
    Args:
        exp_dir: Path to the experiment directory
        
    Returns:
        DataFrame with timestep, factor_vae_eval_accuracy, dci_disentanglement columns
    """
    metrics_dir = os.path.join(exp_dir, METRICS_FOLDER)
    
    if not os.path.exists(metrics_dir):
        print(f"  Warning: metrics_sin folder not found in {exp_dir}")
        return None
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(metrics_dir, '*.json'))
    
    if not json_files:
        print(f"  Warning: No JSON files found in {metrics_dir}")
        return None
    
    records = []
    for json_path in json_files:
        # Extract timestep from filename (e.g., "3750.json" -> 3750)
        filename = os.path.basename(json_path)
        timestep_str = os.path.splitext(filename)[0]
        
        try:
            timestep = int(timestep_str)
        except ValueError:
            print(f"  Warning: Cannot parse timestep from {filename}, skipping")
            continue
        
        metrics = extract_metrics_from_json(json_path)
        records.append({
            'timestep': timestep,
            'factor_vae_eval_accuracy': metrics['factor_vae_eval_accuracy'],
            'dci_disentanglement': metrics['dci_disentanglement']
        })
    
    # Create DataFrame and sort by timestep
    df = pd.DataFrame(records)
    df = df.sort_values('timestep').reset_index(drop=True)
    
    return df


def main():
    """Main function to process all experiment directories."""
    print("=" * 60)
    print("Timestep Metrics Summary Generator")
    print("=" * 60)
    
    for exp_dir in EXPERIMENT_DIRS:
        exp_name = os.path.basename(exp_dir)
        print(f"\nProcessing: {exp_name}")
        print("-" * 50)
        
        if not os.path.exists(exp_dir):
            print(f"  Error: Directory not found: {exp_dir}")
            continue
        
        df = process_experiment_dir(exp_dir)
        
        if df is None or df.empty:
            print(f"  No data to save for {exp_name}")
            continue
        
        # Filter to keep only timesteps in top K for either FactorVAE or DCI
        top_factor_vae = set(df.nlargest(TOP_K, 'factor_vae_eval_accuracy')['timestep'])
        top_dci = set(df.nlargest(TOP_K, 'dci_disentanglement')['timestep'])
        top_timesteps = top_factor_vae | top_dci  # Union of both
        
        df_filtered = df[df['timestep'].isin(top_timesteps)].copy()
        df_filtered = df_filtered.sort_values('timestep').reset_index(drop=True)
        
        # Save to CSV in the experiment directory
        output_path = os.path.join(exp_dir, OUTPUT_FILENAME)
        df_filtered.to_csv(output_path, index=False)
        
        print(f"  Saved: {output_path}")
        print(f"  Total timesteps (original): {len(df)}")
        print(f"  Total timesteps (filtered top {TOP_K}): {len(df_filtered)}")
        
        # Show summary statistics (on filtered data)
        print(f"\n  Summary Statistics (filtered):")
        print(f"  {'Metric':<30} {'Min':>10} {'Max':>10} {'Mean':>10}")
        print(f"  {'-'*60}")
        
        for col in ['factor_vae_eval_accuracy', 'dci_disentanglement']:
            valid = df_filtered[col].dropna()
            if len(valid) > 0:
                print(f"  {col:<30} {valid.min():>10.4f} {valid.max():>10.4f} {valid.mean():>10.4f}")
            else:
                print(f"  {col:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        
        # Show best timesteps (from original data)
        print(f"\n  Best Timesteps:")
        for col, name in [('factor_vae_eval_accuracy', 'FactorVAE'), 
                          ('dci_disentanglement', 'DCI')]:
            valid_df = df.dropna(subset=[col])
            if len(valid_df) > 0:
                best_idx = valid_df[col].idxmax()
                best_timestep = valid_df.loc[best_idx, 'timestep']
                best_value = valid_df.loc[best_idx, col]
                print(f"  {name:<15} best at timestep {best_timestep:>8} (value: {best_value:.4f})")
            else:
                print(f"  {name:<15} N/A")
    
    print("\n" + "=" * 60)
    print("Done! CSV summary files created in each experiment directory.")
    print("=" * 60)


if __name__ == '__main__':
    main()
