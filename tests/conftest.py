"""Shared fixtures.

The suite never touches the checked-in data/orders.db: it generates a fresh corpus into a
temp directory and points DB_PATH at that, so tests cannot corrupt real data and do not
depend on whether scripts/make_corpus.py has been run.
"""

from __future__ import annotations

import os
import random
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set before anything instantiates Settings. Deliberately unreachable endpoints: no test
# may depend on a network service being up.
os.environ.setdefault("HF_TOKEN", "test-token-not-real")
os.environ.setdefault("QDRANT_URL", "http://qdrant.invalid:6333")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("QDRANT_COLLECTION", "test_collection")


@pytest.fixture(scope="session", autouse=True)
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A generated corpus + database for the whole session."""
    from app.config import get_settings
    from scripts.make_corpus import build_db, write_docs

    data_dir = tmp_path_factory.mktemp("desicart-data")
    write_docs(data_dir / "docs")
    build_db(data_dir / "orders.db", random.Random(4412))

    os.environ["DB_PATH"] = str(data_dir / "orders.db")
    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def docs_dir(corpus: Path) -> Path:
    return corpus / "docs"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server() -> Iterator[str]:
    """The real app on a real port.

    The agent's MCP client speaks HTTP, so an in-process ASGI shortcut would test a
    different code path than the one that runs in production.
    """
    import uvicorn

    from app import main

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True, name="test-uvicorn")
    thread.start()

    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("test server died during startup")
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("test server did not start within 30s")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=15)
