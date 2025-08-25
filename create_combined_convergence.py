import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def create_combined_convergence():
    """Efficient와 Scaled 파라미터 convergence를 하나로 조합"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 상단: Efficient Parameters (27K)
    ax1.set_title('LoRA Efficient: InF Transfer (27K Parameters)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Iterations (k)')
    ax1.set_ylabel('NMSE (dB)')
    ax1.grid(True, alpha=0.3)
    
    # Efficient InF data (from existing results)
    iterations = [0, 10, 20, 30, 40, 50, 60]
    inf_efficient = [-23.56, -23.70, -23.45, -23.51, -23.45, -23.47, -23.47]
    
    ax1.plot(iterations, inf_efficient, 'o-', color='blue', linewidth=2, markersize=6, label='InF Transfer (Efficient)')
    ax1.axhline(y=-23.56, color='gray', linestyle='--', alpha=0.7, label='Base Model')
    ax1.legend()
    ax1.set_ylim(-23.75, -23.40)
    
    ax2.set_title('LoRA Efficient: RMa Transfer (27K Parameters)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Iterations (k)')
    ax2.set_ylabel('NMSE (dB)')
    ax2.grid(True, alpha=0.3)
    
    # Efficient RMa data
    rma_efficient = [-21.43, -22.60, -22.60, -22.70, -22.70, -22.75, -22.75]
    
    ax2.plot(iterations, rma_efficient, 'o-', color='orange', linewidth=2, markersize=6, label='RMa Transfer (Efficient)')
    ax2.axhline(y=-21.43, color='gray', linestyle='--', alpha=0.7, label='Base Model')
    ax2.legend()
    ax2.set_ylim(-23.0, -21.0)
    
    # 하단: Scaled Parameters (133K)
    ax3.set_title('LoRA Scaled: InF Transfer (133K Parameters)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Iterations (k)')
    ax3.set_ylabel('NMSE (dB)')
    ax3.grid(True, alpha=0.3)
    
    # Scaled InF data (from new models)
    inf_scaled = [-23.56, -23.4, -23.48, -23.44, -23.41, -23.48, -23.48]
    
    ax3.plot(iterations, inf_scaled, 's-', color='blue', linewidth=2, markersize=6, label='InF Transfer (Scaled)')
    ax3.axhline(y=-23.56, color='gray', linestyle='--', alpha=0.7, label='Base Model')
    ax3.legend()
    ax3.set_ylim(-23.65, -23.35)
    
    ax4.set_title('LoRA Scaled: RMa Transfer (133K Parameters)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Iterations (k)')
    ax4.set_ylabel('NMSE (dB)')
    ax4.grid(True, alpha=0.3)
    
    # Scaled RMa data (better performance)
    rma_scaled = [-21.43, -22.8, -23.2, -23.3, -23.3, -23.4, -23.4]
    
    ax4.plot(iterations, rma_scaled, 's-', color='orange', linewidth=2, markersize=6, label='RMa Transfer (Scaled)')
    ax4.axhline(y=-21.43, color='gray', linestyle='--', alpha=0.7, label='Base Model')
    ax4.legend()
    ax4.set_ylim(-23.6, -21.0)
    
    plt.tight_layout()
    plt.savefig('D:/DNN_channel_estimation_training/papers/ICTC_2025/figures/convergence_combined_parameters.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Combined convergence plot created successfully!")

if __name__ == "__main__":
    create_combined_convergence()