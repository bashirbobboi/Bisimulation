# tests/test_app_smoke.py
"""
Smoke test for the Streamlit app: verifies that the app imports and runs without error.
"""
import pytest
import streamlit as st


def test_app_imports_and_runs(monkeypatch):
    """Test that the Streamlit app can be imported and run without exceptions."""
    # Prevent Streamlit from opening a browser or gathering usage stats
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    # Monkey-patch st.run to a no-op if it exists (for compatibility)
    monkeypatch.setattr(st, "run", lambda *args, **kwargs: None, raising=False)
    # Import the app; if this succeeds, the test passes
    import app

    # If that succeeds with no exceptions, you've got a passing smoke test.
