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
    page_title="KDC-Chat PDF",
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
Sei un assistente esperto che risponde a domande sui documenti. 
Usa le seguenti informazioni estratte dal documento per rispondere alla domanda dell'utente.

ISTRUZIONI IMPORTANTI:
1. Rispondi SEMPRE nella lingua della domanda.
2. La tua risposta deve includere DUE PARTI:
   - Prima parte: informazioni direttamente presenti nel contesto fornito
   - Seconda parte: aggiungi conoscenze rilevanti che non sono esplicitamente menzionate nel contesto ma che arricchiscono la risposta

3. Formatta la tua risposta così:
   - Contenuto dal documento: [risposta basata solo sul documento]
   - *Informazioni aggiuntive: [conoscenze aggiuntive in corsivo]*

4. Se non trovi informazioni nel contesto, dillo chiaramente ma poi offri comunque conoscenze generali sull'argomento.
5. Mantieni un tono professionale ma accessibile.
6. IMPORTANTE: Quando nel contesto sono menzionate immagini tra parentesi quadre [Immagine: ...], considera che sono collegate al testo che le precede. 
   Le descrizioni di immagini sono relative al contenuto del documento e forniscono informazioni visive correlate.

Domanda: {question}
Contesto: {context}
Risposta:
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
    <h1>🏢 Sistema di Analisi Documenti PDF</h1>
    <h3>RAG_AI_SAP_ZOB</h3>
</div>
""", unsafe_allow_html=True)
# FINE BLOCCO CSS



# Verifica se Ollama è disponibile prima di inizializzare i modelli
@st.cache_resource
def init_models():
    try:
        import requests
        import json
        
        st.info("🔍 DEBUG: Inizio inizializzazione modelli...")
        
        # Verifica connessione Ollama
        st.info("🔌 DEBUG: Verifico connessione a Ollama...")
        response = requests.get("http://localhost:11434/api/version", timeout=10)
        if response.status_code == 200:
            version_data = response.json()
            st.info(f"📡 DEBUG: Ollama risponde, versione: {version_data.get('version')}")
        else:
            st.error(f"❌ DEBUG: Ollama risponde con status code: {response.status_code}")
            raise ConnectionError("Ollama non risponde")
        
        # Test del modello per generazione testo
        st.info("🧠 DEBUG: Test del modello gemma3:4b...")
        test_prompt = "test"
        st.info(f"📤 DEBUG: Invio prompt a gemma3:4b: '{test_prompt}'")
        
        test_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b", 
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048
                }
            },
            timeout=30
        )
        
        if test_response.status_code == 200:
            response_data = test_response.json()
            st.info(f"📥 DEBUG: Risposta modello (primi 50 caratteri): {response_data.get('response', '')[:50]}...")
        else:
            st.error(f"❌ DEBUG: Test fallito con status code: {test_response.status_code}")
            if test_response.status_code == 404:
                st.info("Il modello gemma3:4b non è installato. Installalo con:")
                st.code("ollama pull gemma3:4b", language="bash")
            return None, None, None
        
        # Usa nomic-embed-text come modello di embedding principale
        embedding_model = "nomic-embed-text"
        
        # Verifica se nomic-embed-text è disponibile
        st.info(f"🔤 DEBUG: Test del modello embedding: {embedding_model}...")
        embed_response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": embedding_model, 
                "input": "test embedding"
            },
            timeout=30
        )
        
        if embed_response.status_code == 200:
            embed_data = embed_response.json()
            embedding_dim = len(embed_data.get('embedding', []))
            st.info(f"📊 DEBUG: Embedding dimensionalità: {embedding_dim}, primi 5 valori: {embed_data.get('embedding', [])[:5]}...")
        elif embed_response.status_code == 404:
            st.warning(f"❓ DEBUG: Modello {embedding_model} non installato (404)")
            st.info(f"Installazione di {embedding_model}:")
            st.code(f"ollama pull {embedding_model}", language="bash")
            
            # Prova alternative se nomic-embed-text non è disponibile
            backup_models = ["all-minilm", "mxbai-embed-large"]
            
            st.info(f"🔄 DEBUG: Provo modelli di backup: {', '.join(backup_models)}")
            
            for backup_model in backup_models:
                st.info(f"🔍 DEBUG: Test modello alternativo: {backup_model}")
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
                        st.success(f"✅ DEBUG: Modello {backup_model} funziona")
                        embedding_model = backup_model
                        break
                    else:
                        st.warning(f"❌ DEBUG: Modello {backup_model} non funziona (status: {backup_response.status_code})")
                except Exception as e:
                    st.warning(f"❌ DEBUG: Errore con {backup_model}: {str(e)}")
                    
            # Se nessun modello è disponibile
            if embed_response.status_code == 404 and embedding_model == "nomic-embed-text":
                st.error("🚫 DEBUG: Nessun modello di embedding disponibile")
                st.info("Installa il modello di embedding consigliato:")
                st.code("ollama pull nomic-embed-text", language="bash")
                return None, None, None
        else:
            st.error(f"❌ DEBUG: Errore embeddings: {embed_response.status_code}")
            return None, None, None
        
        # Inizializza embeddings
        try:
            st.info(f"🔧 DEBUG: Inizializzazione OllamaEmbeddings con modello: {embedding_model}")
            embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url="http://localhost:11434"
            )
            
            # Test funzionamento
            st.info("🧪 DEBUG: Test embeddings con 'test'...")
            test_result = embeddings.embed_query("test")
            if test_result:
                st.info(f"📊 DEBUG: Embedding generato con dimensione {len(test_result)}, primi 5 valori: {test_result[:5]}")
            else:
                st.error("❌ DEBUG: embed_query ha restituito un risultato vuoto")
                raise Exception("Embeddings restituisce risultato vuoto")
                
        except Exception as e:
            st.error(f"❌ DEBUG: Errore negli embeddings: {e}")
            st.info("Prova a riavviare Ollama:")
            st.code("ollama stop && ollama serve", language="bash")
            return None, None, None
        
        st.info("🗄️ DEBUG: Creazione vector store in memoria...")
        vector_store = InMemoryVectorStore(embeddings)
        
        st.info("🤖 DEBUG: Inizializzazione modello generativo gemma3:4b...")
        model = OllamaLLM(
            model="gemma3:4b",
            base_url="http://localhost:11434",
            num_ctx=2048,
            temperature=0.1
        )
        
        st.success("✅ DEBUG: Setup completo del sistema RAG")
        st.info(f"🧠 Modello generativo: gemma3:4b")
        st.info(f"🔍 Modello embedding: {embedding_model}")
        
        return embeddings, vector_store, model
        
    except requests.exceptions.ConnectionError:
        st.error("🚫 DEBUG: ConnectionError - Impossibile connettersi a Ollama")
        st.code("ollama serve", language="bash")
        return None, None, None
    except Exception as e:
        st.error(f"🚫 DEBUG: Errore generale: {str(e)}")
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

def extract_text(file_path, context=""):
    """Wrapper pubblico per l'estrazione del testo con contesto"""
    # Se c'è contesto, includilo nel prompt
    if context:
        prompt = f"Questa immagine appare nel seguente contesto:\n{context}\n\nDescrivi cosa vedi nell'immagine, considerando il contesto fornito."
    else:
        prompt = "Descrivi brevemente cosa vedi in questa immagine, concentrandoti sul testo e sugli elementi chiave."
    
    try:
        model_with_image_context = model.bind(images=[file_path])
        return model_with_image_context.invoke(prompt)
    except Exception as e:
        return f"Errore nell'elaborazione immagine {file_path}: {str(e)}"

def process_images_parallel(image_files, figures_dir, context_map=None, max_workers=3, timeout_per_image=300):
    """
    Elabora le immagini in parallelo con controlli avanzati e contesto
    
    Args:
        image_files: Lista di file immagine
        figures_dir: Directory delle figure
        context_map: Dizionario che mappa nomi file a contesti
        max_workers: Numero massimo di worker paralleli
        timeout_per_image: Timeout in secondi per ogni immagine
    """
    
    st.info(f"🚀 Elaborazione parallela di {len(image_files)} immagini con {max_workers} worker...")
    
    # Stato di elaborazione per tracciare le immagini (limitato alle prime 20 immagini per evitare loop)
    display_limit = min(20, len(image_files))
    processing_status = {}
    for i, filename in enumerate(image_files[:display_limit]):
        processing_status[filename] = {"status": "pending", "start_time": None, "duration": None}
    
    if len(image_files) > display_limit:
        st.info(f"⚠️ Mostro lo stato solo per le prime {display_limit} immagini (su {len(image_files)} totali)")
    
    # Contenitore per i risultati
    results = []
    
    def process_single_image(file_info):
        index, filename = file_info
        file_path = os.path.join(figures_dir, filename)
        
        # Ottieni il contesto per questa immagine, se disponibile
        context = context_map.get(filename, "") if context_map else ""
        
        start_time = time.time()
        
        # Aggiorna stato solo se è tra quelli visualizzati
        if filename in processing_status:
            processing_status[filename]["status"] = "processing"
            processing_status[filename]["start_time"] = start_time
        
        try:
            # Verifica dimensione file
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limite
                if filename in processing_status:
                    processing_status[filename]["status"] = "skipped"
                return {
                    'index': index,
                    'filename': filename,
                    'text': f"Immagine {filename} troppo grande ({file_size/1024/1024:.1f}MB), saltata",
                    'time': 0,
                    'success': False
                }
            
            extracted_text = extract_text(file_path, context)
            process_time = time.time() - start_time
            
            # Aggiorna stato
            if filename in processing_status:
                processing_status[filename]["status"] = "completed"
                processing_status[filename]["duration"] = process_time
            
            # Se c'è contesto, aggiungi un prefisso esplicito
            if context:
                extracted_text = f"[Nel contesto: {context.strip()[:100]}...]\n{extracted_text}"
            
            return {
                'index': index,
                'filename': filename,
                'text': extracted_text,
                'time': process_time,
                'success': True
            }
        except Exception as e:
            # Aggiorna stato
            if filename in processing_status:
                processing_status[filename]["status"] = "failed"
                processing_status[filename]["duration"] = time.time() - start_time
            
            return {
                'index': index,
                'filename': filename,
                'text': f"Errore: {str(e)}",
                'time': time.time() - start_time,
                'success': False
            }
    
    # Prepara contenitori per il monitoraggio
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Crea una singola area di monitoraggio fissa (non verrà ricreata a ogni ciclo)
    monitoring_expander = st.expander("Monitoraggio elaborazione immagini")
    
    # Crea contenitori fissi per il monitoraggio all'interno dell'expander
    with monitoring_expander:
        monitoring_table = st.empty()
    
    # Crea executor con timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepara i task
        image_tasks = [(i, filename) for i, filename in enumerate(image_files)]
        
        # Memorizza i future per poterli cancellare se necessario
        future_to_image = {executor.submit(process_single_image, task): task for task in image_tasks}
        
        completed = 0
        total_images = len(image_files)
        stuck_warning_shown = {}  # Dizionario per tracciare i warning per file
        
        # Loop principale con monitoraggio
        while future_to_image and completed < total_images:
            # Controlla quali future sono completati
            just_completed = []
            for future in list(future_to_image.keys()):
                if future.done():
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        # Aggiorna progress bar
                        progress = completed / total_images
                        progress_bar.progress(progress)
                        
                        if result['success']:
                            status_text.success(f"✅ {completed}/{total_images}: {result['filename']} elaborata in {result['time']:.2f}s")
                        else:
                            status_text.error(f"❌ {completed}/{total_images}: Errore in {result['filename']}")
                    except Exception as e:
                        completed += 1
                        status_text.error(f"❌ Errore imprevisto: {str(e)}")
                    
                    # Rimuovi dalla lista dei future attivi
                    just_completed.append(future)
            
            # Rimuovi i future completati
            for future in just_completed:
                del future_to_image[future]
            
            # Aggiorna la tabella di monitoraggio - solo una volta per ciclo
            status_data = []
            for filename, status in processing_status.items():
                if status["status"] == "processing":
                    current_duration = time.time() - status["start_time"] if status["start_time"] else 0
                    
                    # Avvisa su immagini bloccate
                    if current_duration > timeout_per_image and filename not in stuck_warning_shown:
                        with monitoring_expander:
                            st.warning(f"⚠️ L'elaborazione di {filename} sta impiegando molto tempo ({current_duration:.0f}s)")
                            force_key = f"force_{filename.replace('.','_')}"
                            if st.button("Forza completamento", key=force_key):
                                # Trova il future corrispondente e cancellalo
                                for f, (_, fname) in future_to_image.items():
                                    if fname == filename:
                                        f.cancel()
                                        break
                        stuck_warning_shown[filename] = True
                else:
                    current_duration = status["duration"] if status["duration"] else 0
                
                status_data.append({
                    "File": filename,
                    "Stato": status["status"],
                    "Tempo (s)": f"{current_duration:.1f}"
                })
            
            # Aggiorna la tabella di monitoraggio con i dati raccolti
            monitoring_table.table(status_data)
            
            # Controlla timeout
            for future, (index, filename) in list(future_to_image.items()):
                if filename in processing_status and processing_status[filename]["status"] == "processing" and processing_status[filename]["start_time"]:
                    duration = time.time() - processing_status[filename]["start_time"]
                    if duration > timeout_per_image:
                        status_text.warning(f"⚠️ Timeout per {filename} dopo {duration:.1f} secondi. Cancellazione...")
                        
                        # Cancella il future
                        future.cancel()
                        
                        # Aggiungi errore ai risultati
                        results.append({
                            'index': index,
                            'filename': filename,
                            'text': f"Timeout dopo {duration:.1f} secondi",
                            'time': duration,
                            'success': False
                        })
                        
                        # Aggiorna stato e rimuovi dai future attivi
                        if filename in processing_status:
                            processing_status[filename]["status"] = "timeout"
                            processing_status[filename]["duration"] = duration
                        del future_to_image[future]
                        
                        completed += 1
                        progress_bar.progress(completed / total_images)
            
            # Pausa per non sovraccaricare l'interfaccia
            time.sleep(0.5)
        
        # Risultati finali
        status_text.success(f"✅ Completato: {completed}/{total_images} immagini elaborate")
        
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
        
        # Organizza gli elementi mantenendo l'ordine e le relazioni
        organized_elements = []
        context_window = []
        
        for element in elements:
            # Se è un'immagine, mantieni il contesto precedente
            if element.category in ["Image", "Table"]:
                # Prendi fino a 3 elementi di testo precedenti come contesto
                previous_context = "\n".join([e.text for e in context_window[-3:] if hasattr(e, 'text')])
                
                # Crea un elemento contestualizzato
                contextualized_element = {
                    "type": element.category,
                    "content": element.text if hasattr(element, 'text') else "",
                    "filename": element.metadata.filename if hasattr(element, 'metadata') and hasattr(element.metadata, 'filename') else None,
                    "context": previous_context,
                    "original_index": len(organized_elements)
                }
                organized_elements.append(contextualized_element)
            else:
                # Elementi di testo normali
                organized_elements.append({
                    "type": element.category,
                    "content": element.text,
                    "context": "",
                    "original_index": len(organized_elements)
                })
                # Aggiorna la finestra di contesto
                context_window.append(element)
                if len(context_window) > 5:  # Mantieni gli ultimi 5 elementi
                    context_window.pop(0)
    
        return organized_elements
        
    except Exception as e:
        # Cattura e mostra eventuali errori in modo più leggibile
        error_message = str(e)
        if "TesseractNotFoundError" in error_message:
            st.error("🚫 Tesseract OCR non è installato. Per installarlo su macOS, esegui:")
            st.code("brew install tesseract", language="bash")
        else:
            st.error(f"Errore durante l'elaborazione del PDF: {error_message}")
        return f"Errore durante l'elaborazione del PDF: {error_message}"

def split_text(text, element_data=None):
    st.info("✂️ Inizio suddivisione del testo in chunks...")
    
    # Se abbiamo dati sugli elementi, usa una strategia più intelligente
    if element_data:
        chunks = []
        current_chunk = ""
        current_context = ""
        
        for element in element_data:
            if element["type"] in ["Image", "Table"]:
                # Se il chunk corrente è troppo grande, dividilo
                if len(current_chunk) > 800:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Aggiorna il contesto corrente
                current_context = element.get("context", "")
                
                # Aggiungi prefisso per chiarire la relazione
                image_text = f"[Immagine correlata al testo precedente: {element['content']}]"
                
                # Aggiungi al chunk corrente o crea un nuovo chunk
                if current_chunk:
                    current_chunk += "\n\n" + image_text
                else:
                    current_chunk = current_context + "\n\n" + image_text
            else:
                # Testo normale
                current_chunk += "\n" + element["content"]
                
                # Se il chunk diventa troppo grande, dividilo
                if len(current_chunk) > 1000:
                    chunks.append(current_chunk)
                    # Mantieni un po' di overlap
                    current_chunk = current_chunk[-200:]
        
        # Aggiungi l'ultimo chunk se non è vuoto
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    else:
        # Usa il metodo standard se non abbiamo i dati degli elementi
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        chunks = text_splitter.split_text(text)
        return chunks

def index_docs(texts):
    if vector_store is None:
        st.error("🚫 Vector store non disponibile. Verifica la connessione a Ollama.")
        return False
    
    st.info(f"🗃️ DEBUG: Inizio indicizzazione di {len(texts)} chunks...")
    start_time = time.time()
    
    try:
        # Indicizza in batch più piccoli per evitare errori
        batch_size = 5
        total_texts = len(texts)
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            batch_start = time.time()
            
            st.info(f"📦 DEBUG: Inizio processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")
            st.info(f"📝 DEBUG: Campione testo del batch: '{batch[0][:100]}...'")
            
            try:
                st.info("🧮 DEBUG: Creazione embeddings per il batch...")
                # Prima creiamo gli embeddings
                batch_embeddings = embeddings.embed_documents(batch)
                st.info(f"📊 DEBUG: Embeddings creati, dimensione: {len(batch_embeddings)}x{len(batch_embeddings[0])}")
                
                # Poi aggiungiamo al vector store
                st.info("🗄️ DEBUG: Aggiunta al vector store...")
                vector_store.add_texts(batch)
                
                batch_time = time.time() - batch_start
                st.success(f"✅ DEBUG: Batch {i//batch_size + 1} completato in {batch_time:.2f}s")
            except Exception as batch_error:
                st.error(f"❌ DEBUG: Errore nel batch {i//batch_size + 1}: {str(batch_error)}")
                st.info("Prova a riavviare Ollama se l'errore persiste")
                return False
        
        index_time = time.time() - start_time
        st.success(f"✅ DEBUG: {len(texts)} documenti indicizzati in {index_time:.2f}s")
        
        # Verifica il vector store
        st.info("🔍 DEBUG: Verifica vector store con query di test...")
        test_results = vector_store.similarity_search("test query", k=1)
        st.info(f"✓ DEBUG: Vector store risponde, {len(test_results)} risultato trovato")
        
        # Salva i chunks elaborati
        st.session_state.processed_chunks = texts
        return True
        
    except Exception as e:
        st.error(f"❌ DEBUG: Errore durante l'indicizzazione: {str(e)}")
        import traceback
        st.error(f"Stack trace: {traceback.format_exc()}")
        st.info("Possibili soluzioni:")
        st.code("ollama stop && ollama serve", language="bash")
        st.code("ollama pull gemma3:4b", language="bash")
        return False

def retrieve_docs(query):
    if vector_store is None:
        st.error("🚫 Vector store non disponibile. Verifica la connessione a Ollama.")
        return []
    
    st.info(f"🔎 DEBUG: Inizio ricerca per query: '{query}'")
    start_time = time.time()
    
    try:
        # Mostra il processo di embedding della query
        st.info("🔡 DEBUG: Creo embedding della query...")
        query_vector = embeddings.embed_query(query)
        st.info(f"📊 DEBUG: Query embedding creato, dimensione: {len(query_vector)}, primi 5 valori: {query_vector[:5]}")
        
        st.info("🔍 DEBUG: Eseguo similarity search nel vector store...")
        docs = vector_store.similarity_search(query)
        search_time = time.time() - start_time
        st.success(f"✅ DEBUG: Ricerca completata in {search_time:.2f}s, trovati {len(docs)} documenti")
        
        # Mostra dettagli sui documenti trovati e i loro punteggi
        with st.expander(f"📋 Documenti rilevanti trovati ({len(docs)})"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Documento {i+1}**")
                st.text(f"Contenuto: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.text(f"Metadata: {doc.metadata}")
        
        return docs
    except Exception as e:
        st.error(f"❌ DEBUG: Errore durante la ricerca: {str(e)}")
        return []

def answer_question(question, documents):
    if model is None:
        st.error("🚫 Modello non disponibile. Verifica la connessione a Ollama.")
        return "Errore: Modello non disponibile"
    
    st.info("🤖 DEBUG: Inizio generazione risposta...")
    start_time = time.time()
    
    try:
        context = "\n\n".join([doc.page_content for doc in documents])
        st.info(f"📝 DEBUG: Contesto preparato, lunghezza: {len(context)} caratteri")
        
        # Mostra il prompt completo
        prompt_template = ChatPromptTemplate.from_template(template)
        formatted_prompt = prompt_template.format(question=question, context=context[:500] + "..." if len(context) > 500 else context)
        st.info("📨 DEBUG: Prompt formattato per il modello:")
        with st.expander("👀 Visualizza prompt completo"):
            st.code(formatted_prompt, language="markdown")
        
        st.info("🚀 DEBUG: Invio prompt al modello gemma3:4b...")
        chain = prompt_template | model
        
        # Timestamp prima dell'invocazione
        invoke_time = time.time()
        st.info(f"⏱️ DEBUG: Timestamp pre-invocazione: {invoke_time}")
        
        response = chain.invoke({"question": question, "context": context})
        
        # Timestamp dopo la risposta
        response_time = time.time()
        st.info(f"⏱️ DEBUG: Timestamp post-risposta: {response_time}")
        st.info(f"⏱️ DEBUG: Tempo di risposta LLM: {response_time - invoke_time:.2f}s")
        
        st.info("📥 DEBUG: Risposta ricevuta dal modello:")
        st.code(response, language="markdown")
        
        generation_time = time.time() - start_time
        st.success(f"✅ DEBUG: Generazione completata in {generation_time:.2f} secondi")
        
        return response
    except Exception as e:
        st.error(f"❌ DEBUG: Errore durante la generazione: {str(e)}")
        import traceback
        st.error(f"Stack trace: {traceback.format_exc()}")
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

def get_similar_docs(query, k=3):
    if vector_store is None:
        return []
    
    # Recupera i chunks più simili
    documents = vector_store.similarity_search(query, k=k)
    
    # Se tra i risultati c'è un'immagine, cerca di recuperare anche i chunks adiacenti
    image_indices = []
    for i, doc in enumerate(documents):
        if "[Immagine" in doc.page_content:
            image_indices.append(i)
    
    # Se ci sono immagini, recupera chunks aggiuntivi per contesto
    if image_indices and len(documents) < 5:
        # Recupera alcuni documenti aggiuntivi
        additional_docs = vector_store.similarity_search(query, k=3)
        # Aggiungi solo quelli non già presenti
        for doc in additional_docs:
            if doc not in documents:
                documents.append(doc)
                if len(documents) >= 5:
                    break
    
    return documents