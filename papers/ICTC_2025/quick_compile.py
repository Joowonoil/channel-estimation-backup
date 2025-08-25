import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ {description} - Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description} - Exception: {e}")
        return False

def main():
    print("Quick LaTeX compilation...")
    
    miktex_path = r"C:\Users\calor\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
    pdflatex = os.path.join(miktex_path, "pdflatex.exe")
    
    # Simple compilation
    cmd = f'"{pdflatex}" -interaction=nonstopmode -halt-on-error ICTC_2025_main.tex'
    
    if run_command(cmd, "LaTeX compilation"):
        if os.path.exists("ICTC_2025_main.pdf"):
            size = os.path.getsize("ICTC_2025_main.pdf") / 1024
            print(f"✓ PDF created successfully: {size:.1f} KB")
        else:
            print("✗ PDF file not found despite successful compilation")
    else:
        print("✗ Compilation failed")
        # Try to show log
        if os.path.exists("ICTC_2025_main.log"):
            print("\nLast few lines of log:")
            with open("ICTC_2025_main.log", "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.strip())

if __name__ == "__main__":
    main()