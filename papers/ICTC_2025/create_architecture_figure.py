import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_architecture_comparison():
    # Create more compact architecture diagram with better space utilization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    # Clean colors matching our actual model components
    colors = {
        'transformer': '#2E5984',   # Dark blue for transformer layers
        'adapter': '#D73027',       # Red for adapter modules
        'lora': '#4575B4',          # Blue for LoRA components
        'attention': '#5A9BD4',     # Light blue for attention
        'feedforward': '#70AD47',   # Green for feedforward
        'text': 'white',
        'input': 'lightgray'
    }
    
    # ============ Adapter Architecture (Left) ============
    ax1.set_xlim(0, 9)
    ax1.set_ylim(0, 10)
    ax1.set_title('(a) Adapter Architecture', fontsize=12, fontweight='bold', pad=15)
    
    # Draw arrows first (so they appear behind boxes)
    # Input to ConditionNetwork
    ax1.arrow(4.5, 8.5, 0, -0.4, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)
    
    # ConditionNetwork to attention layers
    ax1.arrow(3.8, 7.2, -1, -0.8, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    ax1.arrow(5.2, 7.2, 1, -0.8, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    
    # Attention to FFN
    ax1.arrow(2.25, 5.8, 1.5, -0.9, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    ax1.arrow(6.75, 5.8, -1.5, -0.9, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    
    # FFN to Adapters
    ax1.arrow(3.2, 4.5, -1.4, -1.1, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    ax1.arrow(5.8, 4.5, 1.4, -1.1, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    
    # Main path to output
    ax1.arrow(4.5, 4.5, 0, -2.4, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)
    
    # Adapter residual connections
    ax1.arrow(1.8, 3, 1.5, -1, head_width=0.1, head_length=0.08, fc=colors['adapter'], 
              ec=colors['adapter'], linewidth=1.5, alpha=0.8)
    ax1.arrow(7.2, 3, -1.5, -1, head_width=0.1, head_length=0.08, fc=colors['adapter'], 
              ec=colors['adapter'], linewidth=1.5, alpha=0.8)
    
    # Now draw boxes (so they appear on top of arrows)
    # Input Channel Data
    input_rect = patches.Rectangle((2.5, 8.5), 4, 0.8, facecolor=colors['input'], 
                                  edgecolor='black', linewidth=1.5)
    ax1.add_patch(input_rect)
    ax1.text(4.5, 8.9, 'Channel Data', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # ConditionNetwork
    cond_rect = patches.Rectangle((2.5, 7.2), 4, 0.8, facecolor=colors['transformer'], 
                                 edgecolor='black', linewidth=1.5)
    ax1.add_patch(cond_rect)
    ax1.text(4.5, 7.6, 'ConditionNetwork', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=colors['text'])
    
    # Transformer Layer Components
    # Self-Attention
    self_attn_rect = patches.Rectangle((0.5, 5.8), 3.5, 0.9, facecolor=colors['attention'], 
                                      edgecolor='black', linewidth=1.5)
    ax1.add_patch(self_attn_rect)
    ax1.text(2.25, 6.25, 'Self-Attention', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=colors['text'])
    
    # Cross-Attention
    cross_attn_rect = patches.Rectangle((5, 5.8), 3.5, 0.9, facecolor=colors['attention'], 
                                       edgecolor='black', linewidth=1.5)
    ax1.add_patch(cross_attn_rect)
    ax1.text(6.75, 6.25, 'Cross-Attention', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=colors['text'])
    
    # Feedforward
    ffn_rect = patches.Rectangle((2.5, 4.5), 4, 0.9, facecolor=colors['feedforward'], 
                                edgecolor='black', linewidth=1.5)
    ax1.add_patch(ffn_rect)
    ax1.text(4.5, 4.95, 'Feedforward Network', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=colors['text'])
    
    # Adapter modules - bottleneck structure
    adapter1_rect = patches.Rectangle((0.5, 3), 1.8, 0.9, facecolor=colors['adapter'], 
                                     edgecolor='black', linewidth=1.5)
    ax1.add_patch(adapter1_rect)
    ax1.text(1.4, 3.45, 'Adapter\n(Bottleneck)', ha='center', va='center', 
             fontsize=9, fontweight='bold', color=colors['text'])
    
    adapter2_rect = patches.Rectangle((6.7, 3), 1.8, 0.9, facecolor=colors['adapter'], 
                                     edgecolor='black', linewidth=1.5)
    ax1.add_patch(adapter2_rect)
    ax1.text(7.6, 3.45, 'Adapter\n(Bottleneck)', ha='center', va='center', 
             fontsize=9, fontweight='bold', color=colors['text'])
    
    # Output
    output_rect = patches.Rectangle((2.5, 1.2), 4, 0.8, facecolor='darkgray', 
                                   edgecolor='black', linewidth=1.5)
    ax1.add_patch(output_rect)
    ax1.text(4.5, 1.6, 'Channel Estimation', ha='center', va='center', 
             fontsize=11, fontweight='bold', color=colors['text'])
    
    # ============ LoRA Architecture (Right) ============
    ax2.set_xlim(0, 9)
    ax2.set_ylim(0, 10)
    ax2.set_title('(b) LoRA Architecture', fontsize=12, fontweight='bold', pad=15)
    
    # Draw arrows first (so they appear behind boxes)
    # Input to ConditionNetwork
    ax2.arrow(4.5, 8.5, 0, -0.4, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)
    
    # ConditionNetwork to both paths
    ax2.arrow(3.8, 7.2, -1, -0.8, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=2)
    ax2.arrow(5.2, 7.2, 0.8, -0.6, head_width=0.1, head_length=0.08, fc=colors['lora'], 
              ec=colors['lora'], linewidth=1.5)
    
    # LoRA path: A to B
    ax2.arrow(6.2, 6.2, 0, -0.6, head_width=0.1, head_length=0.08, fc=colors['lora'], 
              ec=colors['lora'], linewidth=1.5)
    
    # Both paths to target modules
    ax2.arrow(2.6, 5.8, 1.5, -1.2, head_width=0.1, head_length=0.08, fc='black', ec='black', linewidth=2)
    ax2.arrow(6.2, 5.2, -1.2, -0.6, head_width=0.1, head_length=0.08, fc=colors['lora'], 
              ec=colors['lora'], linewidth=1.5)
    
    # Target modules to output
    ax2.arrow(4.45, 4.2, 0, -1.8, head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=2)
    
    # Now draw boxes (so they appear on top of arrows)
    # Input Channel Data
    input_rect2 = patches.Rectangle((2.5, 8.5), 4, 0.8, facecolor=colors['input'], 
                                   edgecolor='black', linewidth=1.5)
    ax2.add_patch(input_rect2)
    ax2.text(4.5, 8.9, 'Channel Data', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # ConditionNetwork
    cond_rect2 = patches.Rectangle((2.5, 7.2), 4, 0.8, facecolor=colors['transformer'], 
                                  edgecolor='black', linewidth=1.5)
    ax2.add_patch(cond_rect2)
    ax2.text(4.5, 7.6, 'ConditionNetwork', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=colors['text'])
    
    # Original Weight W₀
    w0_rect = patches.Rectangle((1, 5.8), 3.2, 1.2, facecolor=colors['transformer'], 
                               edgecolor='black', linewidth=2)
    ax2.add_patch(w0_rect)
    ax2.text(2.6, 6.4, 'Original Weight\nW₀', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=colors['text'])
    
    # LoRA Low-Rank Decomposition
    lora_A_rect = patches.Rectangle((5.5, 6.2), 1.4, 0.7, facecolor=colors['lora'], 
                                   edgecolor='black', linewidth=1.5)
    ax2.add_patch(lora_A_rect)
    ax2.text(6.2, 6.55, 'LoRA A\n(r×d)', ha='center', va='center', 
             fontsize=9, fontweight='bold', color=colors['text'])
    
    lora_B_rect = patches.Rectangle((5.5, 5.2), 1.4, 0.7, facecolor=colors['lora'], 
                                   edgecolor='black', linewidth=1.5)
    ax2.add_patch(lora_B_rect)
    ax2.text(6.2, 5.55, 'LoRA B\n(d×r)', ha='center', va='center', 
             fontsize=9, fontweight='bold', color=colors['text'])
    
    # Target modules (attention & FFN layers with LoRA)
    target_rect = patches.Rectangle((1, 4.2), 6.9, 0.8, facecolor=colors['attention'], 
                                   edgecolor='black', linewidth=1.5)
    ax2.add_patch(target_rect)
    ax2.text(4.45, 4.6, 'Target Modules (q_proj, k_proj, v_proj, ffnn_linear)', ha='center', va='center', 
             fontsize=9, fontweight='bold', color=colors['text'])
    
    # Addition symbol
    ax2.text(4.5, 3.2, 'W₀ + α(BA)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Output
    output_rect2 = patches.Rectangle((2.5, 1.2), 4, 0.8, facecolor='darkgray', 
                                    edgecolor='black', linewidth=1.5)
    ax2.add_patch(output_rect2)
    ax2.text(4.5, 1.6, 'Channel Estimation', ha='center', va='center', 
             fontsize=11, fontweight='bold', color=colors['text'])
    
    # Remove all axes and grids for cleaner look
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05, wspace=0.15)  # Reduced spacing between subplots
    
    # Save as high-quality PNG first
    plt.savefig('architecture_comparison_new.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Save as PDF
    plt.savefig('figures/architecture_comparison_new.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close()
    print("Created clean, readable architecture comparison figure")

if __name__ == "__main__":
    create_architecture_comparison()