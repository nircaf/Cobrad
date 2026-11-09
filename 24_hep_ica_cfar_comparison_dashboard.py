"""Dashboard entry point: sleep-stage deltas first, main comparison on demand.

Only the selected page script executes, so the heavy HEP / ICA / CFAR dashboard
does not run until it is picked in the sidebar.
"""
import streamlit as st

st.set_page_config(page_title="HEP · ICA · CFAR", layout="wide")

pages = {
    "Sleep stages": [
        st.Page("hep_dashboard_stage.py", title="Sleep-stage deltas",
                icon="🧠", default=True),
        st.Page("hep_dashboard_main.py", title="HEP · ICA · CFAR comparison",
                icon="🫀"),
    ],
}

st.navigation(pages).run()
