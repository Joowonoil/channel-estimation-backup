import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os

def png_to_pdf(png_path, pdf_path):
    """Convert PNG to PDF while maintaining quality"""
    try:
        # Open the PNG image
        with Image.open(png_path) as img:
            # Convert to RGB if necessary (removes alpha channel)
            if img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = rgb_img
            
            # Save as PDF
            img.save(pdf_path, "PDF", resolution=300.0, quality=95)
        print(f"Converted {png_path} -> {pdf_path}")
        return True
    except Exception as e:
        print(f"Error converting {png_path}: {e}")
        return False

def main():
    # List of clean PNG files to convert
    png_files = [
        "adapter_comparison_clean.png",
        "lora_comparison_clean.png", 
        "convergence_curves_clean.png",
        "cross_domain_performance_clean.png"
    ]
    
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created {figures_dir} directory")
    
    success_count = 0
    for png_file in png_files:
        if os.path.exists(png_file):
            pdf_name = png_file.replace('.png', '.pdf')
            pdf_path = os.path.join(figures_dir, pdf_name)
            
            if png_to_pdf(png_file, pdf_path):
                success_count += 1
        else:
            print(f"Warning: {png_file} not found")
    
    print(f"\nSuccessfully converted {success_count}/{len(png_files)} plots to PDF format")
    print("PDF files saved in figures/ directory")

if __name__ == "__main__":
    main()