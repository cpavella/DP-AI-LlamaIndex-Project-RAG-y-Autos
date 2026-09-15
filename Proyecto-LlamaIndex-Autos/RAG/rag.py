"""
Módulo RAG (Retrieval-Augmented Generation) — versión Qdrant.

Combina lo mejor de dos mundos:
- LlamaParse       -> Document Loader avanzado (parsing con IA: tablas, layout, OCR)
- Qdrant           -> Base de datos vectorial propia (servidor del usuario)

Responsabilidades:
- Ingesta:  documentos → LlamaParse → embeddings → colección en Qdrant
            (ingest_data_with_llamaparse)
- Runtime:  el agente se conecta a la colección existente (load_or_create_index)

Las consultas las maneja el agente en agent.py usando tools/retrieval.py
(index.as_query_engine(...)), compatible con cualquier VectorStoreIndex.

La colección sigue la convención multi-tenant del repo: prefijo `tenant_id_`
con guiones bajos (ver validacion_nombre_tenant_id.py).

Ejecutar la ingesta:
    python RAG/rag.py
"""

import os
import sys
import logging
from pathlib import Path

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from validacion_nombre_tenant_id import validar_qdrant

# Logging claro para ver el proceso de ingesta.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Carga las variables de entorno (.env del proyecto).
load_dotenv()

# --------------------------------------------------------------------------- #
# Variables de entorno
# --------------------------------------------------------------------------- #
# LlamaParse necesita la API key de LlamaCloud (solo para el parsing, ya no
# para el índice: los vectores viven en Qdrant).
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if LLAMA_CLOUD_API_KEY:
    os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY

# Servidor Qdrant propio.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "tenant_id_autos")

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = str(ROOT / "Base_de_Conocimiento")

CHUNK_SIZE = 1024        # Chunks grandes para aprovechar el parsing de LlamaParse
CHUNK_OVERLAP = 200      # Overlap para mantener contexto entre chunks
EMBED_MODEL = "text-embedding-3-small"

# ============================================== Paso 1: Document Loader (LlamaParse) ===============================================
# LlamaParse usa modelos de visión para entender mejor la estructura del PDF
# (tablas, columnas, precios), mucho más preciso que un parser tradicional.
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",   # Devuelve Markdown estructurado, ideal para RAG
    language="es",            # Español para mejor reconocimiento
    verbose=True,
)

# ============================================== Paso 2: Document Splitter ===============================================
Settings.text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ============================================== Paso 3: Embedding Model ===============================================
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)


# --------------------------------------------------------------------------- #
# Precondiciones
# --------------------------------------------------------------------------- #
def verificar_configuracion() -> None:
    """
    Falla rápido y barato: valida credenciales y nombre de colección ANTES de
    cargar documentos o gastar una llamada de embeddings.
    """
    if not QDRANT_URL:
        raise RuntimeError("Falta QDRANT_URL en el .env (URL de tu servidor Qdrant).")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Falta OPENAI_API_KEY en el .env (modelo de embeddings).")

    validacion = validar_qdrant(COLLECTION_NAME)
    if not validacion.ok:
        raise ValueError(
            f"Nombre de colección inválido '{COLLECTION_NAME}': {validacion.motivos}"
        )


# --------------------------------------------------------------------------- #
# Cliente y vector store
# --------------------------------------------------------------------------- #
def get_qdrant_client() -> qdrant_client.QdrantClient:
    """
    Cliente conectado al servidor Qdrant del usuario.

    `port=None` es importante: por defecto qdrant-client añade el puerto 6333
    cuando la URL no lo trae, y eso rompe los servidores detrás de un proxy
    HTTPS (EasyPanel, Coolify, Qdrant Cloud), que escuchan en el 443. Con None
    se respeta el puerto si la URL lo incluye, y si no, se usa el del esquema.
    """
    return qdrant_client.QdrantClient(
        url=QDRANT_URL.rstrip("/"),
        api_key=QDRANT_API_KEY or None,
        port=None,
        timeout=60,
    )


def get_async_qdrant_client() -> qdrant_client.AsyncQdrantClient:
    """Cliente para las consultas asíncronas ejecutadas por FunctionAgent."""
    return qdrant_client.AsyncQdrantClient(
        url=QDRANT_URL.rstrip("/"),
        api_key=QDRANT_API_KEY or None,
        port=None,
        timeout=60,
    )


def get_vector_store(
    client: qdrant_client.QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> QdrantVectorStore:
    """
    Vector store de LlamaIndex apuntando a la colección del tenant.

    Si la colección no existe, QdrantVectorStore la crea automáticamente en la
    primera escritura, con la dimensión que devuelva el modelo de embeddings.
    """
    return QdrantVectorStore(
        client=client or get_qdrant_client(),
        aclient=get_async_qdrant_client(),
        collection_name=collection_name,
    )


def collection_has_data(collection_name: str = COLLECTION_NAME) -> bool:
    """True si la colección ya existe en Qdrant y tiene vectores dentro."""
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        return False
    return (client.count(collection_name).count or 0) > 0


# --------------------------------------------------------------------------- #
# Ingesta (crear/actualizar la colección en Qdrant)
# --------------------------------------------------------------------------- #
def ingest_data_with_llamaparse(
    doc_path: str = DOC_PATH,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """
    Pipeline de ingesta: lee los documentos con LlamaParse, los trocea, genera
    los embeddings y los escribe en la colección de Qdrant.
    """
    logging.info("=" * 80)
    logging.info("🚀 INGESTA RAG CON LLAMAPARSE + QDRANT")
    logging.info("=" * 80)

    verificar_configuracion()
    logging.info(f"🎯 Colección destino: '{collection_name}' en {QDRANT_URL}")

    # ---- Paso 1: Document Loader con LlamaParse ----
    logging.info(f"🔍 Cargando documentos desde '{doc_path}' con LlamaParse...")
    file_extractor = {
        ".pdf": parser,
        ".docx": parser,
        ".pptx": parser,
    }
    documents = SimpleDirectoryReader(
        doc_path,
        recursive=True,
        exclude_hidden=True,
        file_extractor=file_extractor,
    ).load_data()

    if not documents:
        logging.warning("⚠️ No se encontraron documentos para procesar.")
        raise RuntimeError(f"No hay documentos en '{doc_path}'.")

    logging.info(f"✅ {len(documents)} documento(s) cargado(s). Subiendo a Qdrant…")
    logging.info("⏳ LlamaParse puede tardar un poco, pero mejora la calidad.")

    # ---- Paso 2: Trocear + embeddings + escritura en Qdrant ----
    vector_store = get_vector_store(collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    total = get_qdrant_client().count(collection_name).count
    logging.info("=" * 80)
    logging.info(f"🎉 Colección '{collection_name}' lista en Qdrant ({total} vectores).")
    logging.info("=" * 80)
    return index


# --------------------------------------------------------------------------- #
# Conexión (runtime) — usada por el agente
# --------------------------------------------------------------------------- #
def connect_to_qdrant_index(
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """Se conecta a una colección ya poblada en Qdrant (sin re-ingestar)."""
    vector_store = get_vector_store(collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store)


def load_or_create_index(
    doc_path: str = DOC_PATH,
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """
    Devuelve el índice listo para consultar (compatible con agent.py).

    - Si la colección ya tiene vectores → se conecta a ella (rápido, sin costo).
    - Si está vacía o no existe → la crea ingestando los documentos con LlamaParse.
    """
    verificar_configuracion()

    if collection_has_data(collection_name):
        logging.info(f"🔗 Conectando a la colección '{collection_name}' en Qdrant…")
        return connect_to_qdrant_index(collection_name)

    logging.info(f"📥 La colección '{collection_name}' está vacía o no existe.")
    logging.info("📥 Creándola desde documentos con LlamaParse…")
    return ingest_data_with_llamaparse(doc_path, collection_name)


if __name__ == "__main__":
    # Ejecuta este archivo directamente para (re)crear la colección en Qdrant:
    #   python RAG/rag.py
    verificar_configuracion()
    ingest_data_with_llamaparse()
