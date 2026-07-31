"""Streamlit frontend — a thin REST client over the FastAPI backend.

Not imported by anything under ``trustresume.api``/``trustresume.orchestration``
etc.: the dependency points one way (UI -> HTTP -> backend), so the backend
stays usable headless and the ``streamlit`` dependency stays optional (the
``ui`` extra).
"""

from __future__ import annotations
