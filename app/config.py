"""Environment loading and validation. Fails fast, with a readable list of what's missing."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# Where the repo root is, so relative paths in env vars resolve the same way whether the
# app is started from the repo root or from inside the container.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- LLM ---------------------------------------------------------------------------
    # min_length so that an empty value fails validation like a missing one; an unset
    # secret in a deploy UI usually arrives as "" rather than absent.
    hf_token: str = Field(min_length=1, description="Hugging Face token with inference access")
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_model: str = "Qwen/Qwen2.5-72B-Instruct"

    # --- Embeddings --------------------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # Not in the original spec's env list: the OpenAI-compatible router (/v1) has no
    # feature-extraction route, so query embedding needs the hf-inference pipeline host.
    # Kept configurable for the same reason LLM_BASE_URL is.
    embed_base_url: str = "https://router.huggingface.co/hf-inference/models"

    # --- Qdrant ------------------------------------------------------------------------
    qdrant_url: str = Field(min_length=1, description="Qdrant Cloud cluster URL, including port")
    qdrant_api_key: str = ""
    qdrant_collection: str = "desicart_policies"

    # --- App ---------------------------------------------------------------------------
    # Trailing slash matters: the MCP app is mounted at /mcp with an inner path of "/",
    # so the slashless form costs a 307 redirect on every request.
    mcp_server_url: str = "http://127.0.0.1:8000/mcp/"
    db_path: str = "data/orders.db"
    daily_message_cap: int = 500
    per_ip_hourly_cap: int = 10
    max_tool_rounds: int = 4
    request_timeout_s: float = 60.0
    # Prior turns replayed to the model. The client sends the history, so the server holds
    # no session state; this bounds how much of it is trusted per request.
    max_history_turns: int = 6

    @property
    def db_file(self) -> Path:
        path = Path(self.db_path)
        return path if path.is_absolute() else ROOT / path

    @property
    def qdrant_base(self) -> str:
        return self.qdrant_url.rstrip("/")

    @property
    def embed_endpoint(self) -> str:
        return f"{self.embed_base_url.rstrip('/')}/{self.embed_model}/pipeline/feature-extraction"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings. Raises ValidationError; call validate_or_exit() at startup."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment


def validate_or_exit() -> Settings:
    """Startup gate. Prints every missing variable at once rather than one per restart."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        missing = sorted({str(err["loc"][0]).upper() for err in exc.errors()})
        print("Configuration error. These environment variables are missing or invalid:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print("\nCopy .env.example to .env and fill it in.", file=sys.stderr)
        raise SystemExit(1) from exc

    if not settings.db_file.exists():
        print(
            f"Configuration error: DB_PATH points at {settings.db_file}, which does not exist.\n"
            "Run: python scripts/make_corpus.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return settings
