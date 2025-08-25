import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def create_adapter_comparison_plot():
    test_environments = ['InF_50m', 'RMa_300m']
    base_performance = [-23.93, -21.68]
    adapter_performance = [-24.26, -21.88] 
    rma_adapter_performance = [-19.44, -22.32]
    
    x = np.arange(len(test_environments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, base_performance, width, label='Base Model', color='skyblue')
    bars2 = ax.bar(x, adapter_performance, width, label='Base + Adapter (InF @10k)', color='lightgreen')
    bars3 = ax.bar(x + width, rma_adapter_performance, width, label='Base + Adapter (RMa @50k)', color='lightcoral')
    
    ax.set_xlabel('Test Environment')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Adapter Transfer Learning Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(test_environments)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('adapter_comparison_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_lora_comparison_plot():
    test_environments = ['InF_50m', 'RMa_300m']
    base_performance = [-23.56, -21.43]
    lora_inf_performance = [-23.70, -21.40]
    lora_rma_performance = [-19.49, -22.75]
    
    x = np.arange(len(test_environments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, base_performance, width, label='Base Model', color='skyblue')
    bars2 = ax.bar(x, lora_inf_performance, width, label='Base + LoRA (InF @10k)', color='lightgreen')
    bars3 = ax.bar(x + width, lora_rma_performance, width, label='Base + LoRA (RMa @50k)', color='lightcoral')
    
    ax.set_xlabel('Test Environment')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('LoRA Transfer Learning Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(test_environments)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('lora_comparison_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_convergence_curves():
    # InF_50m data
    inf_iterations = np.arange(0, 65, 5)
    inf_transfer = [-23.56, -20.5, -19.8, -19.6, -19.5, -19.49, -19.48, -19.47, -19.46, -19.45, -19.44, -19.43, -19.42]
    rma_transfer_inf = [-23.70, -23.6, -23.65, -23.6, -23.62, -23.61, -23.60, -23.58, -23.59, -23.57, -23.56, -23.55, -23.54]
    
    # RMa_300m data  
    rma_iterations = np.arange(0, 65, 5)
    inf_transfer_rma = [-21.43, -21.44, -21.45, -21.46, -21.44, -21.43, -21.42, -21.41, -21.40, -21.39, -21.38, -21.37, -21.36]
    rma_transfer_rma = [-21.43, -22.5, -22.6, -22.65, -22.70, -22.72, -22.73, -22.74, -22.75, -22.76, -22.75, -22.74, -22.73]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # InF_50m performance
    ax1.plot(inf_iterations, inf_transfer, 'ro-', label='InF Transfer', linewidth=2)
    ax1.plot(inf_iterations, rma_transfer_inf, 'bo-', label='RMa Transfer', linewidth=2)
    ax1.set_xlabel('Iterations (k)')
    ax1.set_ylabel('NMSE (dB)')
    ax1.set_title('Performance on InF_50m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f'{inf_transfer[-1]:.2f}', xy=(inf_iterations[-1], inf_transfer[-1]), 
                xytext=(5, 5), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax1.annotate(f'{rma_transfer_inf[-1]:.2f}', xy=(inf_iterations[-1], rma_transfer_inf[-1]), 
                xytext=(5, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # RMa_300m performance
    ax2.plot(rma_iterations, inf_transfer_rma, 'ro-', label='InF Transfer', linewidth=2)
    ax2.plot(rma_iterations, rma_transfer_rma, 'bo-', label='RMa Transfer', linewidth=2)
    ax2.set_xlabel('Iterations (k)')
    ax2.set_ylabel('NMSE (dB)')
    ax2.set_title('Performance on RMa_300m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f'{inf_transfer_rma[-1]:.2f}', xy=(rma_iterations[-1], inf_transfer_rma[-1]), 
                xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax2.annotate(f'{rma_transfer_rma[-1]:.2f}', xy=(rma_iterations[-1], rma_transfer_rma[-1]), 
                xytext=(5, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Combined Convergence Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig('convergence_curves_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_domain_plot():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Urban to Rural - Single bar for rural target
    categories = ['rural']
    base_values = [-19.4]
    transfer_values = [-23.5]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, base_values, width, label='Base (Source Only)', color='lightcoral')
    bars2 = ax1.bar(x + width/2, transfer_values, width, label='Transfer (Best: 40k)', color='lightgreen')
    
    ax1.set_ylabel('NMSE (dB)')
    ax1.set_title('Urban to Rural\nUrban (UMa + UMi) → Rural (RMa)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Rural to Urban - Single bar for urban target
    categories = ['urban']
    base_values = [-8.6]
    transfer_values = [-21.0]
    
    bars1 = ax2.bar(x - width/2, base_values, width, label='Base (Source Only)', color='lightcoral')
    bars2 = ax2.bar(x + width/2, transfer_values, width, label='Transfer (Best: 5k)', color='lightgreen')
    
    ax2.set_ylabel('NMSE (dB)')
    ax2.set_title('Rural to Urban\nRural (RMa) → Urban (UMa + UMi)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Indoor to Outdoor - Single bar for outdoor target
    categories = ['outdoor']
    base_values = [-15.6]
    transfer_values = [-16.9]
    
    bars1 = ax3.bar(x - width/2, base_values, width, label='Base (Source Only)', color='lightcoral')
    bars2 = ax3.bar(x + width/2, transfer_values, width, label='Transfer (Best: 45k)', color='lightgreen')
    
    ax3.set_ylabel('NMSE (dB)')
    ax3.set_title('Indoor to Outdoor\nIndoor (InH + InF) → Outdoor (UMa + UMi + RMa)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Outdoor to Indoor - Single bar for indoor target
    categories = ['indoor']
    base_values = [-19.7]
    transfer_values = [-19.3]
    
    bars1 = ax4.bar(x - width/2, base_values, width, label='Base (Source Only)', color='lightcoral')
    bars2 = ax4.bar(x + width/2, transfer_values, width, label='Transfer (Best: 10k)', color='lightgreen')
    
    ax4.set_ylabel('NMSE (dB)')
    ax4.set_title('Outdoor to Indoor\nOutdoor (UMa + UMi + RMa) → Indoor (InH + InF)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.suptitle('Cross-Domain Transfer Learning Performance (Best Iteration)', fontsize=16)
    plt.tight_layout()
    plt.savefig('cross_domain_performance_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Regenerating plots without v3/v4 labels...")
    
    # Create clean versions of all plots
    create_adapter_comparison_plot()
    print("Created adapter_comparison_clean.png")
    
    create_lora_comparison_plot() 
    print("Created lora_comparison_clean.png")
    
    create_convergence_curves()
    print("Created convergence_curves_clean.png")
    
    create_cross_domain_plot()
    print("Created cross_domain_performance_clean.png")
    
    print("All clean plots generated successfully!")