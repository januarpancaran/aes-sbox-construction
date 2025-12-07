"""
Main Application View
"""

import pandas as pd
import streamlit as st

from src.pages.affine_matrix_explorer import (AffineMatrixExplorer,
                                              render_affine_matrix_explorer)
from src.pages.results_comparison import (ResultsManager,
                                          render_results_comparison)
from src.pages.sbox_constructor import SBoxConstructor, render_sbox_constructor
from src.pages.sbox_tester import SBoxTester, render_sbox_tester


def main():
    """
    Main application entry point
    """
    st.set_page_config(
        page_title="AES S-box Construction",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("🔐 AES S-box Construction")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Affine Matrix Exploration",
            "📦 S-box Construction",
            "🧪 S-box Testing",
            "📊 Results & Comparison",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
    **Paper Implementation:**
    
    "AES S-box modification uses affine matrices exploration for increased S-box strength"
    
    by Alamsyah et al. (2024)
    """
    )

    # Route to different pages
    if page == "🏠 Home":
        render_home()
    elif page == "🔍 Affine Matrix Exploration":
        render_affine_matrix_explorer()
    elif page == "📦 S-box Construction":
        render_sbox_constructor()
    elif page == "🧪 S-box Testing":
        render_sbox_tester()
    elif page == "📊 Results & Comparison":
        render_results()


def render_home():
    """
    Home page
    """
    st.title("🔐 AES S-box Construction Tool")

    st.markdown(
        """
    ## Welcome to the AES S-box Modification Implementation
    
    This application implements the research paper:
    **"AES S-box modification uses affine matrices exploration for increased S-box strength"**
    
    ### 📋 Overview
    
    This tool allows you to:
    
    1. **🔍 Explore Affine Matrices** - Browse through the 2^64 possible affine matrices
    2. **📦 Construct S-boxes** - Build S-boxes using different affine matrices
    3. **🧪 Test S-boxes** - Evaluate S-boxes against cryptographic criteria
    4. **📊 Compare Results** - Compare with AES and other S-boxes
    
    ### 🎯 Key Features
    
    - Exploration of 18,446,744,073,709,551,616 affine matrices
    - Balance and bijectivity testing
    - Cryptographic strength evaluation (NL, SAC, BIC-NL, BIC-SAC, LAP, DAP)
    - Comparison with original AES S-box
    - Interactive visualizations
    
    ### 🚀 Getting Started
    
    Use the sidebar navigation to explore different sections of the application.
    """
    )

    # Display key metrics from the paper
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Possible Matrices", value="256", delta="2^8 combinations"
        )

    with col2:
        st.metric(label="Input Required", value="First Row Only", delta="8 bits")

    with col3:
        st.metric(label="Matrix Generation", value="Automatic", delta="Circular shift")

    st.markdown("---")

    # Quick access buttons
    st.subheader("⚡ Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Explore Matrices", width="stretch"):
            st.session_state.page = "🔍 Affine Matrix Exploration"
            st.rerun()

    with col2:
        if st.button("📦 Construct S-box", width="stretch"):
            st.session_state.page = "📦 S-box Construction"
            st.rerun()

    with col3:
        if st.button("🧪 Test S-box", width="stretch"):
            st.session_state.page = "🧪 S-box Testing"
            st.rerun()


def render_sbox_construction():
    """
    S-box construction page
    """
    render_sbox_constructor()


def render_sbox_testing():
    """
    S-box testing page
    """
    render_sbox_tester()


def render_results():
    """
    Results and comparison page
    """
    render_results_comparison()


if __name__ == "__main__":
    main()
