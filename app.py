"""AI CV Matching — Baseline Results. Streamlit entry point.

Run from the project root:
    py -m streamlit run "group assignment/app.py"

The five analysis pages are registered explicitly so the sidebar contains exactly
the intended pages and every page shares the same header, data checks and privacy
notes.
"""
from __future__ import annotations

import streamlit as st

from app_utils import setup_page, sidebar

setup_page()
sidebar()

nav = st.navigation([
    st.Page("pages/1_Overview.py", title="Overview", default=True),
    st.Page("pages/2_Keyword_Merging.py", title="Keyword extraction & merging"),
    st.Page("pages/3_Autoencoder.py", title="Autoencoder"),
    st.Page("pages/4_Search_Results.py", title="Search results"),
    st.Page("pages/5_Error_Analysis.py", title="Recall & error analysis"),
])
nav.run()
