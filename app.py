"""
Main Application View
Demonstrates how to import and use the Affine Matrix Explorer component
"""

import streamlit as st
import pandas as pd

# Import the component
from src.components.affine_matrix_explorer import (
    render_affine_matrix_explorer,
    AffineMatrixExplorer,
)


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
        # Call the imported component
        render_affine_matrix_explorer()
    elif page == "📦 S-box Construction":
        render_sbox_construction()
    elif page == "🧪 S-box Testing":
        render_sbox_testing()
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
    S-box construction page (placeholder)
    """
    st.title("📦 S-box Construction")
    st.info(
        "This section will be implemented next. It will handle the construction of S-boxes using selected affine matrices."
    )

    # Example of how to use the explorer class programmatically
    explorer = AffineMatrixExplorer()
    example_matrices = explorer.get_example_matrices()

    st.subheader("Preview: Available Example Matrices for Construction")

    selected = st.selectbox("Select a matrix:", options=list(example_matrices.keys()))

    matrix = example_matrices[selected]
    st.write(f"**{selected}**")

    matrix_df = pd.DataFrame(
        matrix,
        columns=[f"Col {i}" for i in range(8)],
        index=[f"Row {i}" for i in range(8)],
    )
    st.dataframe(matrix_df, width="stretch")


def render_sbox_testing():
    """
    S-box testing page (placeholder)
    """
    st.title("🧪 S-box Testing")
    st.info(
        "This section will implement the cryptographic strength tests: NL, SAC, BIC-NL, BIC-SAC, LAP, and DAP."
    )


def render_results():
    """
    Results and comparison page (placeholder)
    """
    st.title("📊 Results & Comparison")
    st.info(
        "This section will display comparisons between constructed S-boxes, AES S-box, and results from previous studies."
    )


if __name__ == "__main__":
    main()
