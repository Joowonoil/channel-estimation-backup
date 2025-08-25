import matplotlib.pyplot as plt
import numpy as np

def create_clean_performance_plot(data_dict, title, output_path, param_info):
    """adapter_comparison_clean.png 스타일의 깔끔한 플롯 생성"""
    
    # 데이터 설정
    environments = ['InF_50m', 'RMa_300m']
    base_values = [data_dict['base_inf'], data_dict['base_rma']]
    inf_transfer_values = [data_dict['inf_transfer_inf'], data_dict['inf_transfer_rma']]
    rma_transfer_values = [data_dict['rma_transfer_inf'], data_dict['rma_transfer_rma']]
    
    # 플롯 설정
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 바 너비와 위치 설정
    bar_width = 0.8
    x_pos = np.arange(len(environments))
    
    # 색상 설정 (clean 스타일과 동일)
    colors = ['#87CEEB', '#90EE90', '#F08080']  # skyblue, lightgreen, lightcoral
    
    # 바 차트 생성
    bars1 = ax.bar(x_pos, base_values, bar_width, label='Base Model', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x_pos, inf_transfer_values, bar_width, label=f'Base + {param_info} (InF @10k)', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x_pos, rma_transfer_values, bar_width, label=f'Base + {param_info} (RMa @50k)', color=colors[2], alpha=0.8)
    
    # 값 표시
    for i, (base, inf, rma) in enumerate(zip(base_values, inf_transfer_values, rma_transfer_values)):
        # Base 모델 값 (파란색 바 중앙)
        ax.text(i, base - 1, f'{base}', ha='center', va='top', fontweight='bold', fontsize=10)
        # InF Transfer 값 (녹색 바 중앙)  
        ax.text(i, inf - 1, f'{inf}', ha='center', va='top', fontweight='bold', fontsize=10)
        # RMa Transfer 값 (빨간색 바 중앙)
        ax.text(i, rma - 1, f'{rma}', ha='center', va='top', fontweight='bold', fontsize=10)
    
    # 축 설정
    ax.set_xlabel('Test Environment', fontsize=12, fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(environments)
    
    # Y축 범위 설정 (clean 스타일과 동일)
    ax.set_ylim(-25, 0)
    
    # 격자 설정
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 범례 설정 (clean 스타일과 동일 위치)
    legend = ax.legend(loc='center', bbox_to_anchor=(0.7, 0.3), frameon=True, 
                      fancybox=True, shadow=True, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Clean plot saved: {output_path}")

# 데이터 정의 (원본 플롯에서 추출한 값들)
adapter_efficient_data = {
    'base_inf': -23.42, 'inf_transfer_inf': -23.83, 'rma_transfer_inf': -19.77,
    'base_rma': -21.17, 'inf_transfer_rma': -21.28, 'rma_transfer_rma': -21.64
}

adapter_scaled_data = {
    'base_inf': -24.02, 'inf_transfer_inf': -24.20, 'rma_transfer_inf': -18.73,
    'base_rma': -21.06, 'inf_transfer_rma': -21.21, 'rma_transfer_rma': -22.39
}

lora_efficient_data = {
    'base_inf': -23.56, 'inf_transfer_inf': -23.70, 'rma_transfer_inf': -19.49,
    'base_rma': -21.43, 'inf_transfer_rma': -21.40, 'rma_transfer_rma': -22.75
}

lora_scaled_data = {
    'base_inf': -23.56, 'inf_transfer_inf': -23.48, 'rma_transfer_inf': -17.59,
    'base_rma': -21.43, 'inf_transfer_rma': -21.43, 'rma_transfer_rma': -23.40
}

# 플롯 생성
plot_configs = [
    {
        'data': adapter_efficient_data,
        'title': 'Adapter Transfer Learning Performance (Efficient: 20K Parameters)',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_efficient_clean_new.png',
        'param_info': 'Adapter'
    },
    {
        'data': adapter_scaled_data,
        'title': 'Adapter Transfer Learning Performance (Scaled: 131K Parameters)',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/adapter_scaled_clean_new.png',
        'param_info': 'Adapter'
    },
    {
        'data': lora_efficient_data,
        'title': 'LoRA Transfer Learning Performance (Efficient: 27K Parameters)',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_efficient_clean_new.png',
        'param_info': 'LoRA'
    },
    {
        'data': lora_scaled_data,
        'title': 'LoRA Transfer Learning Performance (Scaled: 133K Parameters)',
        'output': 'D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/lora_scaled_clean_new.png',
        'param_info': 'LoRA'
    }
]

# 모든 플롯 생성
for config in plot_configs:
    create_clean_performance_plot(
        config['data'], 
        config['title'], 
        config['output'], 
        config['param_info']
    )

print("\n✅ All 4 clean-style plots created successfully!")