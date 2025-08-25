import matplotlib.pyplot as plt
import numpy as np

def create_adapter_comparison_plot(data, title, output_file):
    """Original clean style adapter comparison plot"""
    test_environments = ['InF_50m', 'RMa_300m']
    base_performance = [data['base_inf'], data['base_rma']]
    adapter_performance = [data['inf_inf'], data['inf_rma']] 
    rma_adapter_performance = [data['rma_inf'], data['rma_rma']]
    
    x = np.arange(len(test_environments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, base_performance, width, label='Base Model', color='skyblue')
    bars2 = ax.bar(x, adapter_performance, width, label='Base + Adapter (InF @10k)', color='lightgreen')
    bars3 = ax.bar(x + width, rma_adapter_performance, width, label='Base + Adapter (RMa @50k)', color='lightcoral')
    
    ax.set_xlabel('Test Environment')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title(title)
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {output_file}")

def create_lora_comparison_plot(data, title, output_file):
    """Original clean style LoRA comparison plot"""
    test_environments = ['InF_50m', 'RMa_300m']
    base_performance = [data['base_inf'], data['base_rma']]
    lora_inf_performance = [data['inf_inf'], data['inf_rma']]
    lora_rma_performance = [data['rma_inf'], data['rma_rma']]
    
    x = np.arange(len(test_environments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, base_performance, width, label='Base Model', color='skyblue')
    bars2 = ax.bar(x, lora_inf_performance, width, label='Base + LoRA (InF @10k)', color='lightgreen')
    bars3 = ax.bar(x + width, lora_rma_performance, width, label='Base + LoRA (RMa @50k)', color='lightcoral')
    
    ax.set_xlabel('Test Environment')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title(title)
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {output_file}")

# Data from original plots
adapter_efficient_data = {
    'base_inf': -23.42, 'inf_inf': -23.83, 'rma_inf': -19.77,
    'base_rma': -21.17, 'inf_rma': -21.28, 'rma_rma': -21.64
}

adapter_scaled_data = {
    'base_inf': -24.02, 'inf_inf': -24.20, 'rma_inf': -18.73,
    'base_rma': -21.06, 'inf_rma': -21.21, 'rma_rma': -22.39
}

lora_efficient_data = {
    'base_inf': -23.56, 'inf_inf': -23.70, 'rma_inf': -19.49,
    'base_rma': -21.43, 'inf_rma': -21.40, 'rma_rma': -22.75
}

lora_scaled_data = {
    'base_inf': -23.56, 'inf_inf': -23.48, 'rma_inf': -17.59,
    'base_rma': -21.43, 'inf_rma': -21.43, 'rma_rma': -23.40
}

if __name__ == "__main__":
    print("Regenerating all plots with original clean style...")
    
    # Create adapter plots
    create_adapter_comparison_plot(
        adapter_efficient_data,
        'Adapter Transfer Learning Performance (Efficient: 20K Parameters)',
        'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_efficient_final.png'
    )
    
    create_adapter_comparison_plot(
        adapter_scaled_data,
        'Adapter Transfer Learning Performance (Scaled: 131K Parameters)',
        'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_scaled_final.png'
    )
    
    # Create LoRA plots
    create_lora_comparison_plot(
        lora_efficient_data,
        'LoRA Transfer Learning Performance (Efficient: 27K Parameters)',
        'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_efficient_final.png'
    )
    
    create_lora_comparison_plot(
        lora_scaled_data,
        'LoRA Transfer Learning Performance (Scaled: 133K Parameters)',
        'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_scaled_final.png'
    )
    
    print("\nAll 4 plots generated successfully with original clean style!")