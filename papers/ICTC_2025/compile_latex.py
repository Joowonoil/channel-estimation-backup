import subprocess
import os
import sys
from pathlib import Path

def find_miktex():
    """Find MiKTeX installation path"""
    possible_paths = [
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'MiKTeX' / 'miktex' / 'bin' / 'x64',
        Path('C:/Program Files/MiKTeX/miktex/bin/x64'),
        Path('C:/Program Files (x86)/MiKTeX/miktex/bin'),
    ]
    
    for path in possible_paths:
        pdflatex_path = path / 'pdflatex.exe'
        if pdflatex_path.exists():
            print(f"Found MiKTeX at: {path}")
            return path
    
    # Try to find using where command
    try:
        result = subprocess.run(['where', 'pdflatex'], capture_output=True, text=True)
        if result.returncode == 0:
            pdflatex_path = result.stdout.strip()
            print(f"Found pdflatex at: {pdflatex_path}")
            return Path(pdflatex_path).parent
    except:
        pass
    
    return None

def compile_latex(tex_file):
    """Compile LaTeX document"""
    # Change to the document directory
    doc_dir = Path(tex_file).parent
    os.chdir(doc_dir)
    
    # Find MiKTeX
    miktex_path = find_miktex()
    
    if miktex_path is None:
        print("ERROR: MiKTeX not found. Please ensure MiKTeX is installed.")
        print("You can install it using: winget install MiKTeX.MiKTeX")
        return False
    
    # Set up commands
    pdflatex_cmd = str(miktex_path / 'pdflatex.exe')
    bibtex_cmd = str(miktex_path / 'bibtex.exe')
    base_name = Path(tex_file).stem
    
    # Compilation steps
    commands = [
        ([pdflatex_cmd, '-interaction=nonstopmode', tex_file], "First LaTeX compilation"),
        ([bibtex_cmd, base_name], "BibTeX processing"),
        ([pdflatex_cmd, '-interaction=nonstopmode', tex_file], "Second LaTeX compilation"),
        ([pdflatex_cmd, '-interaction=nonstopmode', tex_file], "Final LaTeX compilation"),
    ]
    
    for cmd, description in commands:
        print(f"\n{description}...")
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Warning: {description} had issues (return code: {result.returncode})")
                # Don't fail on bibtex errors, they're often warnings
                if 'bibtex' not in cmd[0].lower():
                    print("Error output:")
                    print(result.stderr[:1000] if result.stderr else result.stdout[:1000])
            else:
                print(f"[OK] {description} completed successfully")
                
        except subprocess.TimeoutExpired:
            print(f"ERROR: {description} timed out")
            return False
        except Exception as e:
            print(f"ERROR during {description}: {e}")
            return False
    
    # Check if PDF was created
    pdf_file = Path(base_name + '.pdf')
    if pdf_file.exists():
        print(f"\n[SUCCESS] PDF created: {pdf_file.absolute()}")
        print(f"File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        return True
    else:
        print(f"\nERROR: PDF file not created")
        return False

if __name__ == "__main__":
    # Path to the LaTeX file
    tex_file = "ICTC_2025_main.tex"
    
    print("="*60)
    print("ICTC 2025 Paper LaTeX Compilation")
    print("="*60)
    
    if compile_latex(tex_file):
        print("\nCompilation completed successfully!")
    else:
        print("\nCompilation failed. Please check the errors above.")
        sys.exit(1)