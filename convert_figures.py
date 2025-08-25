"""
Convert actual experimental PNG figures to PDF for paper use
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import shutil

def convert_png_to_pdf(png_path, pdf_path, dpi=300):
    """Convert PNG to PDF with high quality"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load and display image
    img = mpimg.imread(png_path)
    ax.imshow(img)
    ax.axis('off')  # Remove axes
    
    # Save as PDF with high DPI
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Converted: {png_path.name} -> {pdf_path.name}")

def main():
    """Convert key experimental figures for paper"""
    
    # Define figure mappings: experimental PNG -> paper PDF
    figure_mappings = {
        # Main comparison figures
        'v3_adapter_comparison.png': 'performance_comparison.pdf',
        'v4_lora_comparison.png': 'efficiency_scatter.pdf', 
        'optimized_lora_comparison.png': 'resource_comparison.pdf',
        'combined_convergence_curves.png': 'convergence_analysis.pdf',
        'cross_domain_performance.png': 'cross_domain_comparison.pdf'
    }
    
    # Paths
    project_root = Path('D:/DNN_channel_estimation_training')
    figures_dir = project_root / 'papers' / 'ICTC_2025' / 'figures'
    
    print("Converting experimental figures to PDF format...")
    print("=" * 60)
    
    converted_count = 0
    
    for png_name, pdf_name in figure_mappings.items():
        # Find PNG file in project
        png_files = list(project_root.glob(f"**/{png_name}"))
        
        if not png_files:
            print(f"Warning: {png_name} not found")
            continue
            
        png_path = png_files[0]  # Use first match
        pdf_path = figures_dir / pdf_name
        
        # Convert PNG to PDF
        try:
            convert_png_to_pdf(png_path, pdf_path)
            converted_count += 1
            
            # Also copy PNG version for backup
            png_backup = figures_dir / png_name
            shutil.copy2(png_path, png_backup)
            
        except Exception as e:
            print(f"Error converting {png_name}: {e}")
    
    print("=" * 60)
    print(f"Successfully converted {converted_count} figures")
    
    # List all files in figures directory
    print("\nFigures directory contents:")
    for file in sorted(figures_dir.glob("*")):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()