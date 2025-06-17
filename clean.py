# Fix definitivo per l'errore RuntimeError di torch._classes.__path__._path
# Questo deve essere il PRIMO codice eseguito
import sys
import warnings
import os

# Disabilita completamente il watcher per i moduli torch
os.environ["STREAMLIT_WATCHER_BLACKLIST"] = "torch,torch.*,unstructured,unstructured.*"

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Patch più robusta per evitare l'errore di torch._classes
def patch_torch_classes():
    try:
        import torch
        # Crea un mock per __path__ se non esiste
        if hasattr(torch, '_classes'):
            original_getattr = torch._classes.__class__.__getattr__
            
            def safe_getattr(self, name):
                if name == '__path__':
                    # Restituisce un oggetto che emula _path
                    class MockPath:
                        @property
                        def _path(self):
                            return []
                    return MockPath()
                return original_getattr(self, name)
            
            torch._classes.__class__.__getattr__ = safe_getattr
    except Exception:
        pass

patch_torch_classes()

import streamlit as st

import os
import ssl
import subprocess
import logging
import time
from datetime import datetime

import concurrent.futures
from functools import lru_cache

# Configurazione della pagina - DEVE essere la prima chiamata Streamlit
st.set_page_config(
    page_title="Zobele Trento - Chat PDF",
    page_icon="🏢",
    layout="wide"
)

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test di diagnostica per dipendenze
with st.sidebar.expander("Diagnostica Sistema"):
    st.markdown("### Verifica Ollama")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            st.success("✅ Ollama è in esecuzione")
            version_data = response.json()
            st.text(f"Versione: {version_data.get('version', 'N/A')}")
        else:
            st.error("❌ Ollama non risponde correttamente")
    except requests.exceptions.ConnectionError:
        st.error("❌ Ollama non è in esecuzione!")
        st.code("ollama serve", language="bash")
        st.write("Oppure avvia Ollama dall'applicazione desktop.")
    except ImportError:
        st.error("❌ Modulo requests non trovato")
    except Exception as e:
        st.error(f"❌ Errore nella verifica di Ollama: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Verifica Tesseract OCR")
    try:
        tesseract_version = subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT).decode()
        st.success(f"✅ Tesseract OCR installato")
        with st.expander("Dettagli versione"):
            st.text(tesseract_version)
    except FileNotFoundError:
        st.error("❌ Tesseract OCR non trovato! Per installarlo su macOS:")
        st.code("brew install tesseract", language="bash")
    except Exception as e:
        st.error(f"❌ Errore nel verificare Tesseract: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Verifica Poppler")
    
    # Lista dei binari di Poppler da cercare
    poppler_binaries = ["pdfinfo", "pdftoppm", "pdftotext", "pdftocairo"]
    poppler_found = False
    
    # Aggiungi i percorsi corretti per Poppler
    os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:" + os.environ.get("PATH", "")
    os.environ["POPPLER_PATH"] = "/opt/homebrew/bin"
    
    # Verifica la presenza di ciascun binario
    for binary in poppler_binaries:
        try:
            binary_path = subprocess.check_output(["which", binary]).decode().strip()
            st.success(f"✅ {binary}")
            poppler_found = True
        except subprocess.CalledProcessError:
            st.error(f"❌ {binary} non trovato")

# Correzione per errori SSL in NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = '/Users/pietrocaracristi/Desktop/RAG_AI_ZOB/pdfs/'
figures_directory = '/Users/pietrocaracristi/Desktop/RAG_AI_ZOB/figures/'    
# se non esistono, restituisco errore   
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs(figures_directory, exist_ok=True)



# BLOCCO CSS PERSONALIZZATO ZOBELE
st.markdown("""
<style>
/* Tema Zobele - Grigio Scuro e Bianco */
body {
    color: #FFFFFF; /* Testo principale bianco */
    background-color: #374151 !important; /* Sfondo grigio scuro */
}

.stApp {
    background-color: #374151 !important; /* Sfondo principale dell'app grigio scuro */
}

/* Forza sfondo grigio scuro per tutti i contenitori principali */
[data-testid="stAppViewContainer"] {
    background-color: #374151 !important;
}

[data-testid="stHeader"] {
    background-color: #374151 !important;
}

[data-testid="stToolbar"] {
    background-color: #374151 !important;
}

[data-testid="stDecoration"] {
    background-color: #374151 !important;
}

[data-testid="stMainContainer"] {
    background-color: #374151 !important;
}

/* Header personalizzato Zobele */
.stApp > header {
    background-color: #1E40AF; /* Blu Zobele */
    color: white;
}

/* Sidebar con sfondo grigio scuro */
.css-1d391kg {
    background-color: #374151 !important; /* Sfondo grigio scuro per la sidebar */
}

/* Expander nella sidebar */
.css-1d391kg .streamlit-expanderHeader {
    background-color: #374151 !important; /* Sfondo grigio scuro per l'header */
    color: #FFFFFF !important; /* Testo bianco */
    border: 1px solid #4B5563 !important; /* Bordo grigio medio */
}

.css-1d391kg .streamlit-expanderContent {
    background-color: #4B5563 !important; /* Sfondo grigio medio per il contenuto */
    border: 1px solid #4B5563 !important; /* Bordo grigio medio */
    border-top: none !important; /* Rimuove il bordo superiore */
}

/* Alert nella sidebar con sfondi grigi scuri */
.css-1d391kg .stAlert {
    background-color: #4B5563 !important; /* Sfondo grigio medio */
    border-radius: 0.375rem !important; /* Angoli arrotondati */
    margin: 0.25rem 0 !important; /* Margini ridotti */
}

/* Testo nella sidebar */
.css-1d391kg .stMarkdown {
    color: #FFFFFF !important; /* Testo bianco */
}

.css-1d391kg .stText {
    color: #FFFFFF !important; /* Testo bianco */
}

.css-1d391kg h3 {
    color: #FFFFFF !important; /* Titoli bianchi */
}

/* Code blocks nella sidebar */
.css-1d391kg .stCode {
    background-color: #4B5563 !important; /* Sfondo grigio medio */
    border: 1px solid #6B7280 !important; /* Bordo grigio */
    color: #FFFFFF !important; /* Testo bianco */
}

/* Fix per i messaggi di log - testo bianco leggibile */
.stAlert {
    color: #FFFFFF !important; /* Testo bianco per tutti gli alert */
    background-color: #4B5563 !important; /* Sfondo grigio medio */
}

.stAlert p {
    color: #FFFFFF !important; /* Testo bianco per i paragrafi negli alert */
}

.stAlert [data-testid="alertContent"] {
    color: #FFFFFF !important; /* Contenuto degli alert */
}

/* Testo generale nell'app principale */
.main .block-container {
    color: #FFFFFF; /* Testo principale bianco */
    background-color: #374151 !important; /* Sfondo grigio scuro */
}

/* Testo nei container di contenuto */
.stMarkdown {
    color: #FFFFFF !important;
}

.stText {
    color: #FFFFFF !important;
}

/* Stile per i messaggi della chat */
[data-testid="stChatMessage"] {
    background-color: #4B5563; /* Sfondo grigio medio */
    border-radius: 0.8rem; /* Angoli arrotondati */
    border: 2px solid #3B82F6; /* Bordo blu */
    color: #FFFFFF; /* Colore del testo bianco */
    margin: 0.5rem 0;
}

/* Messaggi assistente */
[data-testid="stChatMessage"][data-role="assistant"] {
    background-color: #6B7280; /* Grigio leggermente più chiaro */
    color: #FFFFFF; /* Testo bianco */
    border: 2px solid #60A5FA;
}

.stChatInputContainer textarea {
    background-color: #4B5563; /* Sfondo grigio medio */
    color: #FFFFFF; /* Testo bianco */
    border: 2px solid #60A5FA;
    border-radius: 0.5rem;
}

.stFileUploader label {
    color: #FFFFFF; /* Colore etichetta bianco */
    font-weight: 600;
}

/* Titoli e headers */
h1, h2, h3 {
    color: #FFFFFF !important; /* Testo bianco */
    font-weight: 700;
}

/* Logo/Header area personalizzato */
.main-header {
    background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Header Zobele
st.markdown("""
<div class="main-header">
    <h1>🏢 ZOBELE TRENTO</h1>
    <h3>Sistema di Analisi Documenti PDF</h3>
</div>
""", unsafe_allow_html=True)
# FINE BLOCCO CSS



# Verifica se Ollama è disponibile prima di inizializzare i modelli
@st.cache_resource
def init_models():
    try:
        import requests
        
        # Verifica connessione Ollama
        response = requests.get("http://localhost:11434/api/version", timeout=10)
        if response.status_code != 200:
            raise ConnectionError("Ollama non risponde")
        
        # Test del modello per generazione testo
        test_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b", 
                "prompt": "test", 
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048
                }
            },
            timeout=30
        )
        
        if test_response.status_code != 200:
            st.error(f"Modello generativo non disponibile: {test_response.status_code}")
            if test_response.status_code == 404:
                st.info("Il modello gemma3:4b non è installato. Installalo con:")
                st.code("ollama pull gemma3:4b", language="bash")
            return None, None, None
        
        # Usa nomic-embed-text come modello di embedding principale
        embedding_model = "nomic-embed-text"
        
        # Verifica se nomic-embed-text è disponibile
        embed_response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": embedding_model, 
                "input": "test embedding"
            },
            timeout=30
        )
        
        if embed_response.status_code != 200:
            if embed_response.status_code == 404:
                st.warning(f"Il modello {embedding_model} non è installato.")
                st.info(f"Installazione di {embedding_model}:")
                st.code(f"ollama pull {embedding_model}", language="bash")
                
                # Prova alternative se nomic-embed-text non è disponibile
                backup_models = ["all-minilm", "mxbai-embed-large"]
                
                for backup_model in backup_models:
                    st.info(f"Provo il modello alternativo: {backup_model}")
                    try:
                        backup_response = requests.post(
                            "http://localhost:11434/api/embed",
                            json={
                                "model": backup_model, 
                                "input": "test embedding"
                            },
                            timeout=30
                        )
                        
                        if backup_response.status_code == 200:
                            st.success(f"Usando il modello alternativo: {backup_model}")
                            embedding_model = backup_model
                            break
                        elif backup_response.status_code == 404:
                            st.warning(f"Anche {backup_model} non è installato")
                        else:
                            st.warning(f"Errore con {backup_model}: {backup_response.status_code}")
                    except Exception as e:
                        st.warning(f"Errore con {backup_model}: {str(e)}")
                        
                # Se nessun modello è disponibile
                if embed_response.status_code == 404 and embedding_model == "nomic-embed-text":
                    st.error("Nessun modello di embedding disponibile")
                    st.info("Installa il modello di embedding consigliato:")
                    st.code("ollama pull nomic-embed-text", language="bash")
                    return None, None, None
            else:
                st.error(f"Errore nel servizio embeddings: {embed_response.status_code}")
                return None, None, None
        
        # Inizializza embeddings
        try:
            st.info(f"📊 Inizializzazione embeddings con modello: {embedding_model}")
            embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url="http://localhost:11434"
            )
            
            # Test funzionamento
            test_result = embeddings.embed_query("test")
            if not test_result:
                raise Exception("Embeddings restituisce risultato vuoto")
                
        except Exception as e:
            st.error(f"Errore negli embeddings: {e}")
            st.info("Prova a riavviare Ollama:")
            st.code("ollama stop && ollama serve", language="bash")
            return None, None, None
        
        vector_store = InMemoryVectorStore(embeddings)
        
        model = OllamaLLM(
            model="gemma3:4b",
            base_url="http://localhost:11434",
            num_ctx=2048,
            temperature=0.1
        )
        
        st.success("✅ Modelli inizializzati correttamente")
        st.info(f"🧠 Modello generativo: gemma3:4b")
        st.info(f"🔍 Modello embedding: {embedding_model}")
        
        return embeddings, vector_store, model
        
    except requests.exceptions.ConnectionError:
        st.error("🚫 Impossibile connettersi a Ollama. Assicurati che sia in esecuzione:")
        st.code("ollama serve", language="bash")
        return None, None, None
    except Exception as e:
        st.error(f"🚫 Errore nell'inizializzazione dei modelli: {str(e)}")
        st.info("Verifica che Ollama sia in esecuzione e che i modelli siano installati:")
        st.code("ollama serve", language="bash")
        st.code("ollama pull gemma3:4b", language="bash")
        st.code("ollama pull nomic-embed-text", language="bash")
        return None, None, None

# Inizializza i modelli
embeddings, vector_store, model = init_models()

# Inizializza session state per mantenere i dati tra le ricariche
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'vector_store_populated' not in st.session_state:
    st.session_state.vector_store_populated = False
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = []

def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    st.info(f"📁 Caricamento file: {file.name}")
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    st.success(f"✅ File salvato in: {file_path}")
    return file_path

# Ottimizzazione: cache per evitare di riprocessare immagini identiche
@lru_cache(maxsize=100)
def extract_text_cached(file_path_hash):
    """Versione cached dell'estrazione testo per evitare riprocessamenti"""
    return extract_text_internal(file_path_hash)

def extract_text_internal(file_path):
    """Funzione interna per l'estrazione del testo dalle immagini"""
    if model is None:
        return "Errore: Modello non disponibile"
    
    try:
        model_with_image_context = model.bind(images=[file_path])
        # Prompt più specifico e conciso per velocizzare l'elaborazione
        return model_with_image_context.invoke("Describe briefly what you see in this image, focusing on text and key elements.")
    except Exception as e:
        return f"Errore nell'elaborazione immagine {file_path}: {str(e)}"

def extract_text(file_path):
    """Wrapper pubblico per l'estrazione del testo"""
    return extract_text_internal(file_path)

def process_images_parallel(image_files, figures_dir, max_workers=3):
    """Elabora le immagini in parallelo per velocizzare il processo"""
    
    st.info(f"🚀 Elaborazione parallela di {len(image_files)} immagini con {max_workers} worker...")
    
    def process_single_image(file_info):
        index, filename = file_info
        file_path = os.path.join(figures_dir, filename)
        start_time = time.time()
        
        try:
            # Verifica dimensione file per evitare immagini troppo grandi
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limite
                return f"Immagine {filename} troppo grande ({file_size/1024/1024:.1f}MB), saltata"
            
            extracted_text = extract_text(file_path)
            process_time = time.time() - start_time
            
            return {
                'index': index,
                'filename': filename,
                'text': extracted_text,
                'time': process_time,
                'success': True
            }
        except Exception as e:
            return {
                'index': index,
                'filename': filename,
                'text': f"Errore: {str(e)}",
                'time': 0,
                'success': False
            }
    
    # Usa ThreadPoolExecutor per elaborazione parallela
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepara i task
        image_tasks = [(i, filename) for i, filename in enumerate(image_files)]
        
        # Crea progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Sottometti tutti i task
        future_to_image = {executor.submit(process_single_image, task): task for task in image_tasks}
        
        results = []
        completed = 0
        
        # Raccogli i risultati man mano che completano
        for future in concurrent.futures.as_completed(future_to_image):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Aggiorna progress bar
            progress = completed / len(image_files)
            progress_bar.progress(progress)
            
            if result['success']:
                status_text.success(f"✅ {result['filename']} elaborata in {result['time']:.2f}s")
            else:
                status_text.error(f"❌ Errore in {result['filename']}")
        
        # Ordina i risultati per index originale
        results.sort(key=lambda x: x['index'])
        
        return results

def load_pdf(file_path):
    try:
        st.info(f"🔍 Inizio elaborazione PDF: {file_path}")
        start_time = time.time()
        
        # Verifica se Tesseract è installato
        try:
            subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT)
            st.success("✅ Tesseract OCR verificato")
        except FileNotFoundError:
            st.error("🚫 Tesseract OCR non è installato. Per installarlo su macOS, esegui:")
            st.code("brew install tesseract", language="bash")
            st.error("Dopo l'installazione, riavvia l'applicazione.")
            return "Errore: Tesseract OCR non è installato. Installalo e riprova."
        
        # Mostra progress bar durante l'elaborazione
        with st.spinner('Elaborazione PDF in corso...'):
            st.info("🔧 Avvio partizionamento PDF con strategia HI_RES...")
            partition_start = time.time()
            
            elements = partition_pdf(
                file_path,
                strategy=PartitionStrategy.HI_RES,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=figures_directory
            )
            
            partition_time = time.time() - partition_start
            st.success(f"✅ Partizionamento completato in {partition_time:.2f} secondi")
            st.info(f"📊 Elementi estratti: {len(elements)}")
            
            # Analizza i tipi di elementi trovati
            element_types = {}
            text_elements = []
            
            for element in elements:
                element_type = element.category
                element_types[element_type] = element_types.get(element_type, 0) + 1
                
                if element.category not in ["Image", "Table"]:
                    text_elements.append(element.text)
            
            # Mostra statistiche degli elementi
            st.info("📋 Tipi di elementi trovati:")
            for elem_type, count in element_types.items():
                st.text(f"  - {elem_type}: {count}")
            
            st.info(f"📝 Elementi di testo estratti: {len(text_elements)}")
            
            # Elabora eventuali immagini estratte
            if os.path.exists(figures_directory):
                image_files = [f for f in os.listdir(figures_directory) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_files:
                    st.info(f"🖼️ Trovate {len(image_files)} immagini da elaborare")
                    
                    # Determina numero ottimale di worker
                    max_workers = min(3, len(image_files), os.cpu_count() or 1)
                    st.info(f"🔧 Uso {max_workers} worker paralleli per l'elaborazione")
                    
                    image_start = time.time()
                    
                    # Elaborazione parallela con controllo delle risorse
                    if len(image_files) > 1:
                        results = process_images_parallel(image_files, figures_directory, max_workers)
                        
                        # Aggiungi i testi estratti
                        successful_extractions = 0
                        for result in results:
                            if result['success']:
                                text_elements.append(result['text'])
                                successful_extractions += 1
                        
                        image_time = time.time() - image_start
                        st.success(f"✅ {successful_extractions}/{len(image_files)} immagini elaborate in {image_time:.2f} secondi")
                        st.info(f"⚡ Velocità media: {image_time/len(image_files):.2f}s per immagine")
                    else:
                        # Singola immagine - elaborazione sequenziale
                        for file in image_files:
                            st.info(f"🔍 Elaborazione immagine: {file}")
                            extracted_text = extract_text(os.path.join(figures_directory, file))
                            text_elements.append(extracted_text)
                        
                        image_time = time.time() - image_start
                        st.success(f"✅ Immagine elaborata in {image_time:.2f} secondi")
                else:
                    st.info("ℹ️ Nessuna immagine trovata nella cartella figures")

        final_text = "\n\n".join(text_elements)
        total_time = time.time() - start_time
        
        st.success(f"🎉 Elaborazione PDF completata in {total_time:.2f} secondi totali")
        st.info(f"📏 Lunghezza testo finale: {len(final_text)} caratteri")
        
        # Mostra anteprima del testo estratto
        with st.expander("👀 Anteprima testo estratto (primi 500 caratteri)"):
            st.text(final_text[:500] + "..." if len(final_text) > 500 else final_text)
        
        return final_text
        
    except Exception as e:
        # Cattura e mostra eventuali errori in modo più leggibile
        error_message = str(e)
        if "TesseractNotFoundError" in error_message:
            st.error("🚫 Tesseract OCR non è installato. Per installarlo su macOS, esegui:")
            st.code("brew install tesseract", language="bash")
        else:
            st.error(f"Errore durante l'elaborazione del PDF: {error_message}")
        return f"Errore durante l'elaborazione del PDF: {error_message}"

def split_text(text):
    st.info("✂️ Inizio suddivisione del testo in chunks...")
    start_time = time.time()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    chunks = text_splitter.split_text(text)
    split_time = time.time() - start_time
    
    st.success(f"✅ Testo suddiviso in {len(chunks)} chunks in {split_time:.2f} secondi")
    
    # Mostra statistiche sui chunks
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    
    st.info(f"📊 Statistiche chunks:")
    st.text(f"  - Numero totale: {len(chunks)}")
    st.text(f"  - Lunghezza media: {avg_length:.0f} caratteri")
    st.text(f"  - Lunghezza min: {min(chunk_lengths) if chunk_lengths else 0}")
    st.text(f"  - Lunghezza max: {max(chunk_lengths) if chunk_lengths else 0}")
    
    return chunks

def index_docs(texts):
    if vector_store is None:
        st.error("🚫 Vector store non disponibile. Verifica la connessione a Ollama.")
        return False
    
    st.info("🗃️ Inizio indicizzazione dei documenti nel vector store...")
    start_time = time.time()
    
    try:
        # Indicizza in batch più piccoli per evitare errori
        batch_size = 5
        total_texts = len(texts)
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            try:
                vector_store.add_texts(batch)
                st.info(f"✅ Batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size} indicizzato")
            except Exception as batch_error:
                st.error(f"❌ Errore nel batch {i//batch_size + 1}: {str(batch_error)}")
                st.info("Prova a riavviare Ollama se l'errore persiste")
                return False
        
        index_time = time.time() - start_time
        st.success(f"✅ {len(texts)} documenti indicizzati con successo in {index_time:.2f} secondi")
        
        # Salva i chunks elaborati
        st.session_state.processed_chunks = texts
        return True
        
    except Exception as e:
        st.error(f"❌ Errore durante l'indicizzazione: {str(e)}")
        st.info("Possibili soluzioni:")
        st.code("ollama stop && ollama serve", language="bash")
        st.code("ollama pull gemma3:4b", language="bash")
        return False

def retrieve_docs(query):
    if vector_store is None:
        st.error("🚫 Vector store non disponibile. Verifica la connessione a Ollama.")
        return []
    
    st.info(f"🔎 Ricerca documenti per: '{query}'")
    start_time = time.time()
    
    try:
        docs = vector_store.similarity_search(query)
        search_time = time.time() - start_time
        st.success(f"✅ Trovati {len(docs)} documenti rilevanti in {search_time:.2f} secondi")
        
        # Mostra anteprima dei documenti trovati
        with st.expander(f"📋 Documenti rilevanti trovati ({len(docs)})"):
            for i, doc in enumerate(docs):
                st.text(f"Documento {i+1}: {doc.page_content[:200]}...")
        
        return docs
    except Exception as e:
        st.error(f"❌ Errore durante la ricerca: {str(e)}")
        return []

def answer_question(question, documents):
    if model is None:
        st.error("🚫 Modello non disponibile. Verifica la connessione a Ollama.")
        return "Errore: Modello non disponibile"
    
    st.info("🤖 Generazione risposta con il modello AI...")
    start_time = time.time()
    
    try:
        context = "\n\n".join([doc.page_content for doc in documents])
        st.info(f"📝 Lunghezza contesto: {len(context)} caratteri")
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        response = chain.invoke({"question": question, "context": context})
        
        generation_time = time.time() - start_time
        st.success(f"✅ Risposta generata in {generation_time:.2f} secondi")
        
        return response
    except Exception as e:
        st.error(f"❌ Errore durante la generazione della risposta: {str(e)}")
        return f"Errore: {str(e)}"

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    if embeddings is None or vector_store is None or model is None:
        st.error("🚫 Sistema non pronto. Verifica che Ollama sia in esecuzione.")
        st.info("Comandi per riavviare Ollama:")
        st.code("ollama stop", language="bash")
        st.code("ollama serve", language="bash")
        st.stop()
    
    # Controlla se il file è cambiato
    file_changed = (st.session_state.current_file_name != uploaded_file.name)
    
    if file_changed or not st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown("## 📋 Log di Elaborazione")
        
        # Log timestamp
        st.info(f"🕐 Inizio elaborazione: {datetime.now().strftime('%H:%M:%S')}")
        
        # Reset dello stato se il file è cambiato
        if file_changed:
            st.session_state.pdf_processed = False
            st.session_state.vector_store_populated = False
            st.session_state.chat_history = []
            st.session_state.processed_chunks = []
            st.session_state.current_file_name = uploaded_file.name
        
        file_path = upload_pdf(uploaded_file)
        text = load_pdf(file_path)
        
        if not text.startswith("Errore"):
            chunked_texts = split_text(text)
            
            # Prova l'indicizzazione
            success = index_docs(chunked_texts)
            
            if success:
                # Salva lo stato
                st.session_state.pdf_processed = True
                st.session_state.vector_store_populated = True
                
                st.markdown("---")
                st.success("🎉 PDF elaborato con successo! Puoi ora fare domande sul contenuto.")
            else:
                st.error("❌ Errore nell'indicizzazione. Il PDF è stato elaborato ma le domande potrebbero non funzionare.")
                st.session_state.pdf_processed = True
                st.session_state.vector_store_populated = False
        else:
            st.error(text)
            st.session_state.pdf_processed = False
            st.session_state.vector_store_populated = False
    else:
        # Il PDF è già stato elaborato
        st.markdown("---")
        st.success(f"📚 PDF '{uploaded_file.name}' già elaborato e pronto per le domande!")
    
    # Mostra la cronologia chat se esiste
    if st.session_state.chat_history:
        st.markdown("### 💬 Cronologia Chat")
        for chat_item in st.session_state.chat_history:
            st.chat_message("user").write(chat_item["question"])
            st.chat_message("assistant").write(chat_item["answer"])
    
    # Input per le domande (solo se il PDF è stato elaborato)
    if st.session_state.pdf_processed and st.session_state.vector_store_populated:
        question = st.chat_input("Fai una domanda sul PDF...")

        if question:
            # Aggiungi la nuova domanda alla chat
            st.chat_message("user").write(question)
            
            # Log della domanda
            st.info(f"❓ Domanda ricevuta: '{question}'")
            
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            
            # Mostra la risposta
            st.chat_message("assistant").write(answer)
            
            # Salva nella cronologia
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })
            
            # Ricarica per mostrare la cronologia aggiornata
            st.rerun()
    else:
        st.warning("⚠️ Carica e elabora un PDF prima di fare domande.")