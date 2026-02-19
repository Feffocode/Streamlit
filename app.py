"""
Questo script Streamlit permette di:
1. Caricare uno o più file PDF
2. Estrarre il testo da ciascun PDF utilizzando PyPDF2.
3. Suddividere il testo estratto in chunk
4. Generare embeddings (all-MiniLM-L6-v2)
5. Memorizzare i chunk in un vector store Chroma persistente (su disco)
6. Ricevere domande dall'utente via chat e rispondere riportando
   il chunk più pertinente estratto dal vector store.
7. Monitorare le emissioni di CO2 con CodeCarbon durante le operazioni
   computazionalmente intensive (embedding).
8. Mostrare una dashboard delle emissioni nella sidebar.
"""

import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from codecarbon import EmissionsTracker

CHROMA_PERSIST_DIR = "./chroma_db"
EMISSIONS_CSV = "emissions.csv"

#  Configurazione pagina 
st.set_page_config(
    page_title="RAG App",
    page_icon="📄",
    layout="wide",
)

#  Inizializzazione sessione 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.title("RAG Application")
st.markdown(
    "Carica uno o più documenti **PDF** dalla sidebar, poi fai domande "
    "nella chat per ottenere il chunk più pertinente dai tuoi documenti."
)

#  Sidebar 
with st.sidebar:
    st.header("📁 Carica i tuoi PDF")
    uploaded_files = st.file_uploader(
        "Seleziona uno o più file PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    process_btn = st.button("Elabora Documenti", type="primary")

    st.divider()

    st.header(" Configurazione Retrieval")
    top_k = st.slider(
        "Numero di chunk da recuperare (k)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Quanti chunk mostrare in risposta alla domanda.",
    )


#  Funzioni di supporto 
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: list[str], embeddings) -> Chroma:
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="pdf_collection",
    )


def load_emissions_data():
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


#  Elaborazione PDF 
if uploaded_files and process_btn:
    with st.spinner("Elaborazione dei PDF in corso..."):
        all_text = ""
        for pdf_file in uploaded_files:
            text = extract_text_from_pdf(pdf_file)
            all_text += text

        chunks = split_text_into_chunks(all_text)

        embedding_model = get_embedding_model()

        tracker = EmissionsTracker(
            project_name="rag_streamlit_embedding",
            measure_power_secs=10,
            save_to_file=True,
            output_file=EMISSIONS_CSV,
            log_level="error",
        )
        tracker.start()
        st.session_state.vector_store = create_vector_store(chunks, embedding_model)
        tracker.stop()

    st.success(f" Elaborati {len(uploaded_files)} file — {len(chunks)} chunk indicizzati.")

#  Interfaccia Chat 
if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("💬 Chatta con i tuoi documenti")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Fai una domanda sui tuoi PDF"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Ricerca del chunk più pertinente..."):
                try:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k},
                    )
                    docs = retriever.invoke(user_question)

                    if not docs:
                        response = "⚠️ Nessun chunk rilevante trovato nei documenti caricati."
                    else:
                        parts = []
                        for i, doc in enumerate(docs, start=1):
                            parts.append(f"**Chunk {i}:**\n\n{doc.page_content}")
                        response = "\n\n---\n\n".join(parts)

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_msg = (
                        f"**Errore durante la ricerca**\n\n"
                        f"**Tipo:** `{type(e).__name__}`\n\n"
                        f"**Dettaglio:** `{e}`"
                    )
                    st.error(error_msg)

#  Dashboard Emissioni (Sidebar) 
with st.sidebar:
    st.divider()
    st.header("🌱 Dashboard Emissioni")
    emissions_data = load_emissions_data()

    if emissions_data is not None:
        if emissions_data["emissions_kg"] > 1:
            emissions_value = f"{emissions_data['emissions_kg']:.6f} kg CO₂eq"
        else:
            emissions_value = f"{emissions_data['emissions_kg'] * 1000:.3f} g CO₂eq"

        st.metric(
            label="Emissioni Totali",
            value=emissions_value,
            help="Quantità totale di CO2 equivalente emessa.",
        )
        st.metric(
            label="Consumo Energetico",
            value=f"{emissions_data['energy_kwh']:.6f} kWh",
            help="Energia totale consumata durante le operazioni.",
        )
        duration_s = emissions_data["duration_s"]
        st.metric(
            label="Durata Calcolo",
            value=f"{duration_s:.1f} s" if duration_s < 60 else f"{duration_s / 60:.1f} min",
            help="Tempo totale di calcolo delle operazioni tracciate.",
        )
        st.metric(
            label="Esecuzioni Tracciate",
            value=emissions_data["n_runs"],
            help="Numero totale di operazioni monitorate da CodeCarbon.",
        )
        with st.expander("Dettagli CSV completo"):
            df = pd.read_csv(EMISSIONS_CSV)
            st.dataframe(
                df[["timestamp", "project_name", "emissions", "energy_consumed", "duration"]],
                use_container_width=True,
            )
    else:
        st.caption(
            "Nessun dato disponibile. Le emissioni verranno "
            "tracciate durante l'elaborazione dei PDF."
        )