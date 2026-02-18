"""
RAG Application - Fase 4: Versione finale con monitoraggio emissioni CO2.

Questo script Streamlit permette di:
1. Caricare uno o più file PDF tramite la sidebar.
2. Estrarre il testo da ciascun PDF utilizzando PyPDF2.
3. Suddividere il testo estratto in chunk (RecursiveCharacterTextSplitter).
4. Generare embeddings con HuggingFace (all-MiniLM-L6-v2).
5. Memorizzare i chunk in un vector store Chroma persistente su disco.
6. Ricevere domande dall'utente via chat e rispondere usando un LLM
   locale (Ollama) con contesto estratto dal vector store.
7. Monitorare le emissioni di CO2 con CodeCarbon durante le operazioni
   computazionalmente intensive (embedding e generazione risposta).
8. Mostrare una dashboard delle emissioni nella sidebar.
"""

import sys
import os

# ──────────────────────────────────────────────
# Workaround: PyTorch è installato in C:\torch_tmp a causa
# della limitazione Windows Long Path. Aggiungiamo il percorso
# al sys.path per renderlo importabile.
# ──────────────────────────────────────────────
_torch_path = r"C:\torch_tmp"
if os.path.isdir(_torch_path) and _torch_path not in sys.path:
    sys.path.insert(0, _torch_path)

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from codecarbon import EmissionsTracker


# ──────────────────────────────────────────────
# Costanti
# ──────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
EMISSIONS_CSV = "emissions.csv"


# ──────────────────────────────────────────────
# Configurazione della pagina Streamlit
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG App – Versione Finale",
    page_icon="📄",
    layout="wide",
)

# ──────────────────────────────────────────────
# Inizializzazione dello stato della sessione
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ──────────────────────────────────────────────
# Titolo principale
# ──────────────────────────────────────────────
st.title("📄 RAG Application – Versione Finale")
st.markdown(
    "Carica uno o più documenti **PDF** dalla sidebar, poi fai domande "
    "nella chat per ottenere risposte basate sul contenuto dei documenti."
)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    # ── Upload dei PDF ──
    st.header("📁 Carica i tuoi PDF")
    uploaded_files = st.file_uploader(
        "Seleziona uno o più file PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Puoi caricare più file contemporaneamente.",
    )

    st.divider()

    # ── Configurazione del modello Ollama ──
    st.header("🤖 Configurazione LLM")
    st.caption("Richiede [Ollama](https://ollama.com) in esecuzione locale.")
    ollama_model = st.text_input(
        "Modello Ollama",
        value="llama3",
        help="Nome del modello Ollama da usare (es. llama3, mistral, phi3).",
    )
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value="http://localhost:11434",
        help="URL del server Ollama.",
    )
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Valori bassi = risposte più precise. Valori alti = più creative.",
    )


# ══════════════════════════════════════════════
# FUNZIONI HELPER
# ══════════════════════════════════════════════


# ──────────────────────────────────────────────
# Funzione: estrazione del testo da un singolo PDF
# ──────────────────────────────────────────────
def extract_text_from_pdf(pdf_file) -> str:
    """
    Legge un file PDF caricato tramite Streamlit e ne restituisce
    tutto il testo concatenato pagina per pagina.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ──────────────────────────────────────────────
# Funzione: chunking del testo con LangChain
# ──────────────────────────────────────────────
def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Suddivide il testo in chunk utilizzando RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


# ──────────────────────────────────────────────
# Funzione: creazione del modello di embedding
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    Carica e restituisce il modello di embedding HuggingFace.
    Utilizza all-MiniLM-L6-v2 (384 dimensioni).
    Cached per evitare ricaricamenti ad ogni rerun.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ──────────────────────────────────────────────
# Funzione: creazione del vector store Chroma
# ──────────────────────────────────────────────
def create_vector_store(chunks: list[str], embeddings) -> Chroma:
    """
    Crea un vector store Chroma persistente su disco a partire
    dai chunk di testo e dal modello di embedding fornito.
    """
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="pdf_collection",
    )


# ──────────────────────────────────────────────
# Funzione: creazione della RAG chain con LCEL
# ──────────────────────────────────────────────
def build_rag_chain(vector_store: Chroma, model_name: str, base_url: str, temp: float):
    """
    Costruisce una RAG chain usando LangChain Expression Language (LCEL).

    Pipeline:
      1. Il retriever cerca i 4 chunk più simili alla domanda.
      2. Il prompt combina il contesto recuperato con la domanda.
      3. L'LLM (Ollama) genera la risposta.
      4. Lo StrOutputParser estrae il testo della risposta.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temp,
    )

    prompt = ChatPromptTemplate.from_template(
        """Sei un assistente AI utile e preciso. Rispondi alla domanda
basandoti ESCLUSIVAMENTE sul contesto fornito di seguito.
Se il contesto non contiene informazioni sufficienti per rispondere,
dillo chiaramente.

Contesto:
{context}

Domanda: {question}

Risposta:"""
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ──────────────────────────────────────────────
# Funzione: lettura e aggregazione emissioni da CSV
# ──────────────────────────────────────────────
def load_emissions_data() -> dict | None:
    """
    Legge il file emissions.csv generato da CodeCarbon e restituisce
    un dizionario con le metriche aggregate:
    - emissions_kg: emissioni totali in kg di CO2eq
    - energy_kwh: consumo energetico totale in kWh
    - duration_s: durata totale del calcolo in secondi
    - n_runs: numero di esecuzioni tracciate

    Ritorna None se il file non esiste.
    """
    if not os.path.isfile(EMISSIONS_CSV):
        return None

    try:
        df = pd.read_csv(EMISSIONS_CSV)
        if df.empty:
            return None

        return {
            "emissions_kg": df["emissions"].sum(),
            "energy_kwh": df["energy_consumed"].sum(),
            "duration_s": df["duration"].sum(),
            "n_runs": len(df),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════
# LOGICA PRINCIPALE
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Elaborazione dei PDF caricati
# ──────────────────────────────────────────────
if uploaded_files:
    with st.spinner("⏳ Elaborazione dei PDF in corso..."):
        all_text = ""
        file_info = []

        for pdf_file in uploaded_files:
            text = extract_text_from_pdf(pdf_file)
            all_text += text
            file_info.append(
                {
                    "name": pdf_file.name,
                    "chars": len(text),
                    "pages": len(PdfReader(pdf_file).pages),
                }
            )

        chunks = split_text_into_chunks(all_text)

    # ── Embedding + Vector Store CON tracking CodeCarbon ──
    with st.spinner("🔄 Embedding e creazione Vector Store (tracking CO2 attivo)..."):
        embedding_model = get_embedding_model()

        # Avvia il tracker CodeCarbon per l'embedding
        tracker = EmissionsTracker(
            project_name="rag_streamlit",
            measure_power_secs=10,
            save_to_file=True,
            output_file=EMISSIONS_CSV,
            log_level="warning",       # evita log verbosi nella console
        )
        tracker.start_task("embedding_vectorstore")

        st.session_state.vector_store = create_vector_store(chunks, embedding_model)

        # Ferma il tracker al termine dell'embedding
        tracker.stop_task()
        tracker.stop()

    # ── Riepilogo elaborazione nella sidebar ──
    with st.sidebar:
        st.divider()
        st.subheader("📊 Stato elaborazione")
        for info in file_info:
            st.write(
                f"📄 **{info['name']}** — "
                f"{info['chars']} car., {info['pages']} pag."
            )
        st.success(f"✅ **{len(chunks)} chunk** nel Vector Store")

else:
    st.info("👈 Carica uno o più file PDF dalla sidebar per iniziare.")

# ──────────────────────────────────────────────
# Interfaccia Chat
# ──────────────────────────────────────────────
if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("💬 Chatta con i tuoi documenti")

    # ── Mostra lo storico dei messaggi ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Input utente ──
    if user_question := st.chat_input("Fai una domanda sui tuoi PDF..."):
        # Aggiungi la domanda allo storico
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # ── Genera la risposta con la RAG chain + tracking CO2 ──
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ricerca e generazione risposta..."):
                try:
                    rag_chain = build_rag_chain(
                        vector_store=st.session_state.vector_store,
                        model_name=ollama_model,
                        base_url=ollama_base_url,
                        temp=temperature,
                    )

                    # Avvia il tracker CodeCarbon per la generazione
                    tracker = EmissionsTracker(
                        project_name="rag_streamlit",
                        measure_power_secs=10,
                        save_to_file=True,
                        output_file=EMISSIONS_CSV,
                        log_level="warning",
                    )
                    tracker.start_task("llm_inference")

                    response = rag_chain.invoke(user_question)

                    # Ferma il tracker al termine della generazione
                    tracker.stop_task()
                    tracker.stop()

                    st.markdown(response)

                    # Salva la risposta nello storico
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    # Tenta di fermare il tracker in caso di errore
                    try:
                        tracker.stop()
                    except Exception:
                        pass

                    error_msg = (
                        f"❌ **Errore nella comunicazione con Ollama.**\n\n"
                        f"Assicurati che:\n"
                        f"1. Ollama sia in esecuzione (`ollama serve`)\n"
                        f"2. Il modello **{ollama_model}** sia installato "
                        f"(`ollama pull {ollama_model}`)\n"
                        f"3. L'URL **{ollama_base_url}** sia corretto\n\n"
                        f"Dettaglio errore: `{e}`"
                    )
                    st.error(error_msg)


# ══════════════════════════════════════════════
# DASHBOARD EMISSIONI (Sidebar)
# ══════════════════════════════════════════════
with st.sidebar:
    st.divider()
    st.header("🌱 Dashboard Emissioni")

    emissions_data = load_emissions_data()

    if emissions_data is not None:
        # ── Emissioni totali ──
        st.metric(
            label="🏭 Emissioni Totali",
            value=f"{emissions_data['emissions_kg']:.6f} kg CO₂eq",
            help="Quantità totale di CO2 equivalente emessa.",
        )

        # ── Consumo energetico ──
        st.metric(
            label="⚡ Consumo Energetico",
            value=f"{emissions_data['energy_kwh']:.6f} kWh",
            help="Energia totale consumata durante le operazioni.",
        )

        # ── Durata calcolo ──
        duration_min = emissions_data["duration_s"] / 60
        st.metric(
            label="⏱️ Durata Calcolo",
            value=f"{emissions_data['duration_s']:.1f} s"
            if emissions_data["duration_s"] < 60
            else f"{duration_min:.1f} min",
            help="Tempo totale di calcolo delle operazioni tracciate.",
        )

        # ── Numero esecuzioni tracciate ──
        st.metric(
            label="📈 Esecuzioni Tracciate",
            value=emissions_data["n_runs"],
            help="Numero totale di operazioni monitorate da CodeCarbon.",
        )

        # ── Link al CSV completo (espandibile) ──
        with st.expander("📄 Dettagli CSV completo"):
            df = pd.read_csv(EMISSIONS_CSV)
            st.dataframe(
                df[["timestamp", "project_name", "emissions", "energy_consumed", "duration"]],
                use_container_width=True,
            )
    else:
        st.caption(
            "Nessun dato disponibile. Le emissioni verranno "
            "tracciate durante l'elaborazione dei PDF e le query."
        )
