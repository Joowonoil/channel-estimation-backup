#!/usr/bin/env python3
"""
Modify existing plot images to remove v3/v4 labels and add characteristics
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

def modify_plot_title(image_path, new_title, output_path):
    """
    Read an existing plot image and modify its title
    """
    # Read the image
    img = mpimg.imread(image_path)
    
    # Create figure with same size as original
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Display the original image
    ax.imshow(img)
    ax.axis('off')
    
    # Add white rectangle to cover old title area (approximately)
    title_rect = Rectangle((0, 0), img.shape[1], int(img.shape[0] * 0.12), 
                          facecolor='white', edgecolor='none')
    ax.add_patch(title_rect)
    
    # Add new title
    ax.text(img.shape[1]/2, int(img.shape[0] * 0.06), new_title, 
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax.transData)
    
    # Save the modified image
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Modified {image_path} -> {output_path}")

def create_all_clean_plots():
    """Create all 8 clean performance plots"""
    
    # Define the mapping of original to clean plots with characteristics
    plot_modifications = [
        # 0.27% Parameter versions (efficient)
        {
            'input': 'v3_adapter_comparison.png',
            'output': 'adapter_comparison_efficient_clean.png',
            'title': 'Adapter Transfer Learning Performance (Efficient: 20K Parameters)'
        },
        {
            'input': 'v3_adapter_improvement.png', 
            'output': 'adapter_improvement_efficient_clean.png',
            'title': 'Adapter Performance Improvement (Efficient: 20K Parameters)'
        },
        {
            'input': 'v4_lora_comparison.png',
            'output': 'lora_comparison_efficient_clean.png', 
            'title': 'LoRA Transfer Learning Performance (Efficient: 27K Parameters)'
        },
        {
            'input': 'v4_lora_improvement.png',
            'output': 'lora_improvement_efficient_clean.png',
            'title': 'LoRA Performance Improvement (Efficient: 27K Parameters)'
        },
        
        # 1.3% Parameter versions (scaled)
        {
            'input': 'v3_adapter_comparison_new_models.png',
            'output': 'adapter_comparison_scaled_clean.png',
            'title': 'Adapter Transfer Learning Performance (Scaled: 131K Parameters)'
        },
        {
            'input': 'v3_adapter_improvement_new_models.png',
            'output': 'adapter_improvement_scaled_clean.png', 
            'title': 'Adapter Performance Improvement (Scaled: 131K Parameters)'
        },
        {
            'input': 'v4_lora_comparison_new_models.png',
            'output': 'lora_comparison_scaled_clean.png',
            'title': 'LoRA Transfer Learning Performance (Scaled: 133K Parameters)'
        },
        {
            'input': 'v4_lora_improvement_new_models.png',
            'output': 'lora_improvement_scaled_clean.png',
            'title': 'LoRA Performance Improvement (Scaled: 133K Parameters)'
        }
    ]
    
    # Process each plot
    for modification in plot_modifications:
        try:
            modify_plot_title(
                modification['input'],
                modification['title'], 
                modification['output']
            )
        except FileNotFoundError:
            print(f"[WARNING] {modification['input']} not found, skipping...")
        except Exception as e:
            print(f"[ERROR] Failed to process {modification['input']}: {e}")

if __name__ == "__main__":
    print("Modifying plot titles to remove v3/v4 labels and add characteristics...")
    create_all_clean_plots()
    print("All clean plots created!")
    print("\nClean plot files:")
    print("- adapter_comparison_efficient_clean.png (20K params)")
    print("- adapter_improvement_efficient_clean.png (20K params)")  
    print("- lora_comparison_efficient_clean.png (27K params)")
    print("- lora_improvement_efficient_clean.png (27K params)")
    print("- adapter_comparison_scaled_clean.png (131K params)")
    print("- adapter_improvement_scaled_clean.png (131K params)")
    print("- lora_comparison_scaled_clean.png (133K params)")  
    print("- lora_improvement_scaled_clean.png (133K params)")