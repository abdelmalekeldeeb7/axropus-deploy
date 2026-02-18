from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "axropus_test.db"
    monkeypatch.setenv("AXROPUS_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AXROPUS_JWT_SECRET", "test-secret")
    monkeypatch.setenv("AXROPUS_TRIAL_KEY_DAYS", "7")
    monkeypatch.setenv("AXROPUS_MAX_KEYS_PER_CUSTOMER", "5")

    for name in list(sys.modules.keys()):
        if name == "backend" or name.startswith("backend."):
            del sys.modules[name]

    main = importlib.import_module("backend.main")
    app = getattr(main, "app")
    with TestClient(app) as test_client:
        yield test_client

