import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

def fix_plot_legend(image_path, new_title, new_legend_text, output_path):
    """완전히 새로운 범례로 플롯을 수정"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(img)
    ax.axis('off')
    
    # 제목 부분 덮기 (상단 12%)
    title_rect = Rectangle((0, 0), img.shape[1], int(img.shape[0] * 0.12), 
                          facecolor='white', edgecolor='none')
    ax.add_patch(title_rect)
    
    # 범례 부분 덮기 (우측 상단)
    legend_rect = Rectangle((img.shape[1] * 0.75, int(img.shape[0] * 0.12)), 
                           img.shape[1] * 0.25, int(img.shape[0] * 0.15), 
                           facecolor='white', edgecolor='none')
    ax.add_patch(legend_rect)
    
    # 새 제목 추가
    ax.text(img.shape[1]/2, int(img.shape[0] * 0.06), new_title, 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 새 범례 추가 (빨간색 박스)
    legend_x = img.shape[1] * 0.85
    legend_y = int(img.shape[0] * 0.18)
    
    # 범례 색상 박스
    legend_color_rect = Rectangle((legend_x - 15, legend_y - 5), 12, 8, 
                                 facecolor='#ff6b6b', edgecolor='black', linewidth=0.5)
    ax.add_patch(legend_color_rect)
    
    # 범례 텍스트
    ax.text(legend_x + 5, legend_y, new_legend_text, 
            ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Fixed plot saved: {output_path}")

# 모든 플롯 수정
plot_configs = [
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_efficient_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_efficient_fixed.png',
        'title': 'Adapter Transfer Learning Performance (Efficient: 20K Parameters)',
        'legend': 'RMa Adapter Transfer'
    },
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_efficient_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_efficient_fixed.png',
        'title': 'LoRA Transfer Learning Performance (Efficient: 27K Parameters)', 
        'legend': 'LoRA RMa Transfer'
    },
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_scaled_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_scaled_fixed.png',
        'title': 'Adapter Transfer Learning Performance (Scaled: 131K Parameters)',
        'legend': 'RMa Adapter Transfer'
    },
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_scaled_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_scaled_fixed.png',
        'title': 'LoRA Transfer Learning Performance (Scaled: 133K Parameters)',
        'legend': 'LoRA RMa Transfer'
    }
]

# 각 플롯 수정
for config in plot_configs:
    try:
        fix_plot_legend(
            config['input'], 
            config['title'], 
            config['legend'], 
            config['output']
        )
    except Exception as e:
        print(f"Error processing {config['input']}: {e}")

print("All plots fixed successfully!")