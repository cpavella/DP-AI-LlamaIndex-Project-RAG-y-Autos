"""
Gestión del histórico de conversación (chat history) en PostgreSQL / Supabase.

Responsabilidades:
- Construir la URL de conexión desde variables de entorno (.env)
- Inicializar el PostgresChatStore (tabla compartida entre todos los usuarios)
- Crear un ChatMemoryBuffer aislado por usuario (chat_id de Telegram)

La configuración (token_limit, table_name) viene de model_config/model.yaml.
Las credenciales de DB vienen del .env.
"""

import os
from urllib.parse import quote_plus

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.storage.chat_store.postgres import PostgresChatStore


def _build_database_url() -> str:
    """Construye la URL de conexión a PostgreSQL."""

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    INSTANCE_UNIX_SOCKET = os.getenv("INSTANCE_UNIX_SOCKET")

    if not all([DB_USER, DB_PASSWORD, DB_NAME]):
        raise ValueError(
            "❌ Faltan variables de base de datos.\n"
            "Requeridas: DB_USER, DB_PASSWORD, DB_NAME"
        )

    # Cloud Run + Cloud SQL
    if INSTANCE_UNIX_SOCKET:
        return (
            f"postgresql+asyncpg://"
            f"{DB_USER}:{quote_plus(DB_PASSWORD)}@/"
            f"{DB_NAME}?host={quote_plus(INSTANCE_UNIX_SOCKET)}"
        )

    # PostgreSQL mediante host/IP
    # Se mantiene para ejecución local o Supabase
    if not DB_HOST:
        raise ValueError(
            "❌ DB_HOST no está definido y tampoco "
            "INSTANCE_UNIX_SOCKET."
        )

    return (
        f"postgresql+asyncpg://"
        f"{DB_USER}:{quote_plus(DB_PASSWORD)}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def build_chat_store(table_name: str) -> PostgresChatStore:
    """
    Crea el PostgresChatStore que persiste todos los mensajes.
    Se crea UNA vez al arrancar el bot y se comparte entre todos los usuarios.
    """
    return PostgresChatStore.from_uri(
        uri=_build_database_url(),
        table_name=table_name,
    )


def get_memory_for_user(
    chat_store: PostgresChatStore,
    chat_id: str,
    token_limit: int,
) -> ChatMemoryBuffer:
    """
    Crea un ChatMemoryBuffer para un usuario específico.
    Cada chat_id (usuario de Telegram) tiene su propia conversación persistida.
    """
    return ChatMemoryBuffer.from_defaults(
        token_limit=token_limit,
        chat_store=chat_store,
        chat_store_key=chat_id,
    )
