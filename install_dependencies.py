import os
import sys
import subprocess
import platform

def print_colored(message, color="green"):
    """Stampa un messaggio colorato nella console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['green'])}{message}{colors['reset']}")

def check_command(command):
    """Verifica se un comando è disponibile nel sistema."""
    try:
        subprocess.check_output(f"which {command}", shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print_colored("=== VERIFICATORE DIPENDENZE PER RAG AI ZOBELE ===", "blue")
    print_colored("\nVerificando le dipendenze di sistema...", "blue")
    
    # Sistema operativo
    system = platform.system()
    print_colored(f"Sistema operativo: {system} {platform.version()}")
    
    # Verifica Tesseract OCR
    if check_command("tesseract"):
        try:
            version = subprocess.check_output("tesseract --version", shell=True).decode().split("\n")[0]
            print_colored(f"✅ Tesseract OCR: {version}")
        except:
            print_colored(f"✅ Tesseract OCR: Installato (versione non determinata)")
    else:
        print_colored("❌ Tesseract OCR: Non installato", "red")
        if system == "Darwin":  # macOS
            print_colored("  Per installare Tesseract OCR su macOS:", "yellow")
            print_colored("  brew install tesseract", "yellow")
        elif system == "Linux":
            print_colored("  Per installare Tesseract OCR su Linux:", "yellow")
            print_colored("  sudo apt-get install tesseract-ocr", "yellow")
        else:
            print_colored("  Scarica Tesseract OCR da: https://github.com/UB-Mannheim/tesseract/wiki", "yellow")
    
    # Verifica Poppler
    poppler_binaries = ["pdfinfo", "pdftoppm", "pdftotext"]
    poppler_found = False
    print_colored("\nVerifica Poppler:")
    for binary in poppler_binaries:
        if check_command(binary):
            try:
                path = subprocess.check_output(f"which {binary}", shell=True).decode().strip()
                print_colored(f"✅ {binary}: {path}")
                poppler_found = True
            except:
                pass
        else:
            print_colored(f"❌ {binary}: Non trovato", "red")
    
    if not poppler_found:
        if system == "Darwin":  # macOS
            print_colored("  Per installare Poppler su macOS:", "yellow")
            print_colored("  brew install poppler", "yellow")
        elif system == "Linux":
            print_colored("  Per installare Poppler su Linux:", "yellow")
            print_colored("  sudo apt-get install poppler-utils", "yellow")
        else:
            print_colored("  Scarica Poppler da: https://poppler.freedesktop.org", "yellow")
    
    # Riepilogo
    print_colored("\n=== RIEPILOGO ===", "blue")
    if not poppler_found or not check_command("tesseract"):
        print_colored("❌ Alcune dipendenze mancano. Installa le dipendenze mancanti e riprova.", "red")
    else:
        print_colored("✅ Tutte le dipendenze di sistema sono installate correttamente!", "green")

if __name__ == "__main__":
    main()
