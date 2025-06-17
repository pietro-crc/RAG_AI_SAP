import os
import sys
import ssl
import subprocess

# Rimozione di Streamlit e uso di print per l'output

print("=== TEST PRESENZA E FUNZIONALITÀ POPPLER ===")

# Mostra il PATH di sistema attuale
print(f"\n=== PATH DI SISTEMA ===")
print(os.environ.get('PATH'))

# Lista dei binari di Poppler da cercare
poppler_binaries = ["pdfinfo", "pdftoppm", "pdftotext", "pdftocairo"]
poppler_found = False

# Verifica la presenza di ciascun binario
print("\n=== BINARI POPPLER ===")
for binary in poppler_binaries:
    try:
        binary_path = subprocess.check_output(["which", binary]).decode().strip()
        print(f"✅ {binary} trovato in: {binary_path}")
        poppler_found = True
    except subprocess.CalledProcessError:
        print(f"❌ {binary} non trovato nel PATH")

# Test funzionale per verificare che Poppler funzioni effettivamente
print("\n=== TEST FUNZIONALE POPPLER ===")
if poppler_found:
    # Verifichiamo la versione di pdfinfo invece di creare un PDF di test
    try:
        result = subprocess.check_output(["pdfinfo", "-v"], stderr=subprocess.STDOUT).decode()
        print("✅ Test funzionale riuscito! pdfinfo ha fornito informazioni sulla versione.")
        print("\nVersione pdfinfo:")
        print(result)
        print("✅ Poppler è installato e funzionante!")
    except subprocess.CalledProcessError as e:
        # Alcuni binari di Poppler potrebbero avere -v o --version
        try:
            result = subprocess.check_output(["pdfinfo", "--version"], stderr=subprocess.STDOUT).decode()
            print("✅ Test funzionale riuscito! pdfinfo ha fornito informazioni sulla versione.")
            print("\nVersione pdfinfo:")
            print(result)
            print("✅ Poppler è installato e funzionante!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Test funzionale fallito! pdfinfo ha restituito un errore:")
            print(e.output.decode() if hasattr(e, 'output') else str(e))
    except Exception as e:
        print(f"❌ Errore durante il test: {str(e)}")
else:
    print("❌ Nessun binario di Poppler trovato, impossibile eseguire test funzionale.")

# Verifica percorsi comuni di installazione
common_paths = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/local/bin"
]

print("\n=== PERCORSI COMUNI ===")
for path in common_paths:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.startswith("pdf")]
        if files:
            print(f"✅ Binari pdf* trovati in {path}: {', '.join(files)}")
        else:
            print(f"📂 Percorso {path} esiste ma non contiene binari pdf*")
    else:
        print(f"❌ Percorso {path} non esiste")

# Aggiunta di percorsi per Poppler (manteniamo questa parte)
os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:" + os.environ.get("PATH", "")
os.environ["POPPLER_PATH"] = "/opt/homebrew/bin"  # Questo è dove si trovano i binari eseguibili

print("\n=== PATH DOPO MODIFICHE ===")
print(os.environ.get("PATH"))
print(f"POPPLER_PATH: {os.environ.get('POPPLER_PATH')}")

# Correzione per errori SSL in NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
