# tests/test_app_smoke.py
import pytest
import streamlit as st

def test_app_imports_and_runs(monkeypatch):
    # Prevent Streamlit from blocking or opening a browser
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    # Monkey-patch st.run to no-op if it exists
    monkeypatch.setattr(st, "run", lambda *args, **kwargs: None, raising=False)
    # Simply import your Streamlit script
    import app 
    # If that succeeds with no exceptions, you’ve got a passing smoke test.
