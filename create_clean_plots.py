#!/usr/bin/env python3
"""
Create clean plots without v3/v4 version labels for paper
"""

import matplotlib.pyplot as plt
import numpy as np

def create_adapter_plot():
    """Create clean Adapter comparison plot"""
    # Data from actual results
    environments = ['InF_50m', 'RMa_300m']
    base_adapter = [-24.02, -21.06]
    inf_adapter = [-24.20, -21.21]  
    rma_adapter = [-18.73, -22.39]
    
    x = np.arange(len(environments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars1 = ax.bar(x - width, base_adapter, width, label='Base (Adapter)', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, inf_adapter, width, label='InF Adapter Transfer', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, rma_adapter, width, label='RMa Adapter Transfer', color=colors[2], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Test Environment', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title('Adapter Transfer Learning Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adapter_comparison_clean_no_version.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved adapter_comparison_clean_no_version.png")
    plt.close()

def create_lora_plot():
    """Create clean LoRA comparison plot"""
    # Data from actual results
    environments = ['InF_50m', 'RMa_300m']
    base_lora = [-23.56, -21.43]
    inf_lora = [-23.48, -21.43]
    rma_lora = [-17.59, -23.40]
    
    x = np.arange(len(environments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars1 = ax.bar(x - width, base_lora, width, label='Base (LoRA)', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, inf_lora, width, label='LoRA (InF @20k)', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, rma_lora, width, label='LoRA (RMa @60k)', color=colors[2], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Test Environment', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title('LoRA Transfer Learning Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lora_comparison_clean_no_version.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved lora_comparison_clean_no_version.png")
    plt.close()

def create_convergence_plot():
    """Create clean convergence plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Data from iteration analysis
    iterations = [0, 10, 20, 30, 40, 50, 60]
    
    # InF Transfer on InF_50m
    inf_on_inf = [-23.56, -23.37, -23.48, -23.44, -23.43, -23.41, -23.38]
    ax1.plot(iterations, inf_on_inf, 'o-', color='#2ecc71', linewidth=2, markersize=6)
    ax1.set_title('InF Transfer on InF_50m', fontsize=12)
    ax1.set_ylabel('NMSE (dB)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-23.56, color='black', linestyle='--', alpha=0.5)
    
    # InF Transfer on RMa_300m  
    inf_on_rma = [-21.43, -21.19, -21.43, -21.23, -21.30, -21.40, -21.28]
    ax2.plot(iterations, inf_on_rma, 'o-', color='#3498db', linewidth=2, markersize=6)
    ax2.set_title('InF Transfer on RMa_300m', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=-21.43, color='black', linestyle='--', alpha=0.5)
    
    # RMa Transfer on InF_50m
    rma_on_inf = [-23.56, -18.99, -18.28, -17.87, -17.67, -17.65, -17.59]
    ax3.plot(iterations, rma_on_inf, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax3.set_title('RMa Transfer on InF_50m', fontsize=12)
    ax3.set_xlabel('Iterations (k)', fontsize=10)
    ax3.set_ylabel('NMSE (dB)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=-23.56, color='black', linestyle='--', alpha=0.5)
    
    # RMa Transfer on RMa_300m
    rma_on_rma = [-21.43, -22.83, -23.16, -23.22, -23.33, -23.39, -23.40]
    ax4.plot(iterations, rma_on_rma, 'o-', color='#f39c12', linewidth=2, markersize=6)
    ax4.set_title('RMa Transfer on RMa_300m', fontsize=12)
    ax4.set_xlabel('Iterations (k)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=-21.43, color='black', linestyle='--', alpha=0.5)
    
    plt.suptitle('LoRA Transfer Learning Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('convergence_analysis_clean_no_version.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved convergence_analysis_clean_no_version.png")
    plt.close()

if __name__ == "__main__":
    print("Creating clean plots without version labels...")
    create_adapter_plot()
    create_lora_plot() 
    create_convergence_plot()
    print("All clean plots created successfully!")