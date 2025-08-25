import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

def fix_plot_completely(image_path, new_title, new_legend_text, output_path):
    """완전히 깨끗한 플롯 생성 - v3/v4 완전 제거"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(img)
    ax.axis('off')
    
    # 제목 부분 완전히 덮기 (상단 15%)
    title_rect = Rectangle((0, 0), img.shape[1], int(img.shape[0] * 0.15), 
                          facecolor='white', edgecolor='none')
    ax.add_patch(title_rect)
    
    # 범례 부분 완전히 덮기 (우측 상단 30%)
    legend_rect = Rectangle((img.shape[1] * 0.7, int(img.shape[0] * 0.15)), 
                           img.shape[1] * 0.3, int(img.shape[0] * 0.25), 
                           facecolor='white', edgecolor='none')
    ax.add_patch(legend_rect)
    
    # 새 제목 추가
    ax.text(img.shape[1]/2, int(img.shape[0] * 0.075), new_title, 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 새 범례 추가
    legend_x = img.shape[1] * 0.75
    legend_y = int(img.shape[0] * 0.25)
    
    # 범례 색상 박스 (빨간색)
    legend_color_rect = Rectangle((legend_x, legend_y), 20, 12, 
                                 facecolor='#ff6b6b', edgecolor='black', linewidth=0.8)
    ax.add_patch(legend_color_rect)
    
    # 범례 텍스트
    ax.text(legend_x + 30, legend_y + 6, new_legend_text, 
            ha='left', va='center', fontsize=11, fontweight='normal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Fixed plot saved: {output_path}")

# 수정할 플롯들
plot_configs = [
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_efficient_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_efficient_final.png',
        'title': 'Adapter Transfer Learning Performance (Efficient: 20K Parameters)',
        'legend': 'Adapter Transfer'
    },
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_scaled_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_comparison_scaled_final.png',
        'title': 'Adapter Transfer Learning Performance (Scaled: 131K Parameters)',
        'legend': 'Adapter Transfer'
    },
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_efficient_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_efficient_final.png',
        'title': 'LoRA Transfer Learning Performance (Efficient: 27K Parameters)', 
        'legend': 'LoRA Transfer'
    },
    {
        'input': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_scaled_clean.png',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_comparison_scaled_final.png',
        'title': 'LoRA Transfer Learning Performance (Scaled: 133K Parameters)',
        'legend': 'LoRA Transfer'
    }
]

# 각 플롯 수정
for config in plot_configs:
    try:
        fix_plot_completely(
            config['input'], 
            config['title'], 
            config['legend'], 
            config['output']
        )
    except Exception as e:
        print(f"Error processing {config['input']}: {e}")

print("All plots completely fixed - v3/v4 totally removed!")