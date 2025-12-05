"""
Affine Matrix Exploration Component for AES S-box Construction
Based on the paper: "AES S-box modification uses affine matrices exploration"

This module generates 8x8 affine matrices using circular right shift pattern.
Only the first row needs to be specified, and subsequent rows are generated automatically.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import time


class AffineMatrixExplorer:
    """
    Class to handle affine matrix exploration for S-box construction.
    Explores 2^8 = 256 possible 8x8 affine matrices based on first row circular shifts.
    """

    def __init__(self):
        self.matrix_size = 8
        self.total_matrices = 2**8  # 256 possible first rows

    def first_row_to_matrix(self, first_row: np.ndarray) -> np.ndarray:
        """
        Generate an 8x8 affine matrix from the first row using circular right shift.
        Each subsequent row is a circular right shift of the previous row.

        Args:
            first_row: 1D numpy array of 8 binary values (the first row)

        Returns:
            8x8 numpy array with binary values
        """
        if len(first_row) != 8:
            raise ValueError("First row must have exactly 8 elements")

        matrix = np.zeros((8, 8), dtype=int)
        matrix[0] = first_row

        # Each row is a circular right shift of the previous row
        for i in range(1, 8):
            matrix[i] = np.roll(matrix[i - 1], 1)

        return matrix

    def index_to_matrix(self, index: int) -> np.ndarray:
        """
        Convert an index (0 to 255) to an 8x8 affine matrix.
        The index represents the first row as an 8-bit number.

        Args:
            index: Integer index representing the first row (0-255)

        Returns:
            8x8 numpy array with binary values
        """
        if index < 0 or index >= self.total_matrices:
            raise ValueError(f"Index must be between 0 and {self.total_matrices - 1}")

        # Convert index to 8-bit binary representation (first row)
        binary_str = format(index, "08b")
        first_row = np.array([int(b) for b in binary_str])

        # Generate full matrix from first row
        return self.first_row_to_matrix(first_row)

    def matrix_to_index(self, matrix: np.ndarray) -> int:
        """
        Convert an 8x8 affine matrix to its index based on the first row.

        Args:
            matrix: 8x8 numpy array with binary values

        Returns:
            Integer index representing the first row (0-255)
        """
        # Get first row and convert to binary string
        first_row = matrix[0]
        binary_str = "".join(str(int(x)) for x in first_row)

        # Convert binary string to integer
        return int(binary_str, 2)

    def validate_affine_structure(self, matrix: np.ndarray) -> bool:
        """
        Validate if the matrix follows the circular shift pattern.

        Args:
            matrix: 8x8 numpy array to validate

        Returns:
            True if matrix follows affine structure, False otherwise
        """
        for i in range(1, 8):
            expected_row = np.roll(matrix[i - 1], 1)
            if not np.array_equal(matrix[i], expected_row):
                return False
        return True

    def get_matrix_range(self, start_idx: int, count: int) -> List[np.ndarray]:
        """
        Get a range of matrices starting from start_idx.

        Args:
            start_idx: Starting index (0-255)
            count: Number of matrices to generate

        Returns:
            List of 8x8 numpy arrays
        """
        matrices = []
        for i in range(count):
            idx = start_idx + i
            if idx < self.total_matrices:
                matrices.append(self.index_to_matrix(idx))
        return matrices

    def get_example_matrices(self) -> dict:
        """
        Get example matrices for demonstration.

        Returns:
            Dictionary containing example matrices
        """
        examples = {}

        # All zeros (index 0)
        examples["All Zeros (Index 0)"] = self.index_to_matrix(0)

        # All ones (index 255)
        examples["All Ones (Index 255)"] = self.index_to_matrix(255)

        # Identity-like pattern (index 1)
        examples["Identity-like (Index 1)"] = self.index_to_matrix(1)

        # Alternating pattern (index 170 = 10101010)
        examples["Alternating (Index 170)"] = self.index_to_matrix(170)

        # Another alternating (index 85 = 01010101)
        examples["Alternating 2 (Index 85)"] = self.index_to_matrix(85)

        return examples


def render_affine_matrix_explorer():
    """
    Streamlit component for exploring affine matrices.
    This function can be imported and called from other views.
    """
    st.header("🔍 Affine Matrix Exploration")

    # Initialize explorer
    explorer = AffineMatrixExplorer()

    # Display information
    st.info(
        f"""
    **Affine Matrix Structure:** Each 8×8 matrix is generated from its first row using circular right shifts.
    
    - **First row:** Can be any 8-bit binary pattern (256 possibilities)
    - **Subsequent rows:** Each row is a circular right shift of the previous row
    - **Total matrices:** 2^8 = {explorer.total_matrices} possible affine matrices
    """
    )

    # Create tabs for different exploration methods
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "✏️ Custom Input",
            "📊 Example Matrices",
            "🔢 Index-based Access",
            "📈 Range Explorer",
        ]
    )

    # Tab 1: Custom Input
    with tab1:
        st.subheader("Create Matrix from First Row")

        st.write("Enter the first row of the matrix (8 binary digits):")

        # Two input methods
        input_method = st.radio(
            "Input method:",
            ["Binary String", "Individual Bits", "Decimal Value"],
            horizontal=True,
        )

        first_row = None

        if input_method == "Binary String":
            binary_input = st.text_input(
                "Enter 8-bit binary string (e.g., 10101010):",
                value="00000001",
                max_chars=8,
                key="binary_string_input",
            )

            if len(binary_input) == 8 and all(c in "01" for c in binary_input):
                first_row = np.array([int(b) for b in binary_input])
            else:
                st.error("Please enter exactly 8 binary digits (0 or 1)")

        elif input_method == "Individual Bits":
            st.write("Toggle each bit (0 or 1):")
            cols = st.columns(8)
            bits = []

            # Use session state to maintain bit values
            if "custom_bits" not in st.session_state:
                st.session_state.custom_bits = [0, 0, 0, 0, 0, 0, 0, 1]

            for i, col in enumerate(cols):
                with col:
                    st.session_state.custom_bits[i] = st.selectbox(
                        f"Bit {i}",
                        options=[0, 1],
                        index=st.session_state.custom_bits[i],
                        key=f"bit_{i}",
                    )

            first_row = np.array(st.session_state.custom_bits)

        else:  # Decimal Value
            decimal_value = st.number_input(
                "Enter decimal value (0-255):",
                min_value=0,
                max_value=255,
                value=1,
                step=1,
                key="decimal_input",
            )
            binary_str = format(decimal_value, "08b")
            first_row = np.array([int(b) for b in binary_str])
            st.info(f"Binary representation: {binary_str}")

        if first_row is not None:
            # Generate matrix
            matrix = explorer.first_row_to_matrix(first_row)
            matrix_index = explorer.matrix_to_index(matrix)

            st.success("✅ Matrix generated successfully!")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Generated Matrix:**")
                st.write("*(Each row is a circular right shift of the previous row)*")

                # Display matrix with row labels
                matrix_df = pd.DataFrame(
                    matrix,
                    columns=[f"Col {i}" for i in range(8)],
                    index=[f"Row {i}" for i in range(8)],
                )
                st.dataframe(matrix_df, width="stretch")

            with col2:
                st.write("**Matrix Properties:**")
                st.metric("Index", matrix_index)
                st.metric("First Row (Binary)", "".join(map(str, first_row)))
                st.metric("First Row (Decimal)", matrix_index)
                st.metric("Rank", np.linalg.matrix_rank(matrix))
                st.metric("Ones Count", int(np.sum(matrix)))
                st.metric("Zeros Count", int(64 - np.sum(matrix)))

                # Check if it's a valid affine structure
                is_valid = explorer.validate_affine_structure(matrix)
                if is_valid:
                    st.success("✓ Valid affine structure")
                else:
                    st.warning("✗ Invalid affine structure")

    # Tab 2: Example Matrices
    with tab2:
        st.subheader("Example Matrices")

        example_matrices = explorer.get_example_matrices()
        matrix_choice = st.selectbox(
            "Select an example matrix:", options=list(example_matrices.keys())
        )

        selected_matrix = example_matrices[matrix_choice]
        matrix_index = explorer.matrix_to_index(selected_matrix)
        first_row = selected_matrix[0]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("**Matrix:**")
            matrix_df = pd.DataFrame(
                selected_matrix,
                columns=[f"Col {i}" for i in range(8)],
                index=[f"Row {i}" for i in range(8)],
            )
            st.dataframe(matrix_df, width="stretch")

        with col2:
            st.write("**Matrix Properties:**")
            st.metric("Index", matrix_index)
            st.metric("First Row (Binary)", "".join(map(str, first_row)))
            st.metric("First Row (Decimal)", matrix_index)
            st.metric("Rank", np.linalg.matrix_rank(selected_matrix))
            st.metric("Ones Count", int(np.sum(selected_matrix)))

    # Tab 3: Index-based Access
    with tab3:
        st.subheader("Access Matrix by Index")

        st.write("Enter an index (0-255) to view the corresponding matrix:")

        # Input method selection
        input_method = st.radio(
            "Input method:",
            ["Decimal", "Binary"],
            horizontal=True,
            key="index_input_method",
        )

        if input_method == "Decimal":
            matrix_idx = st.number_input(
                "Matrix Index (0-255):",
                min_value=0,
                max_value=255,
                value=0,
                step=1,
                key="index_decimal",
            )
        else:
            binary_input = st.text_input(
                "Binary Index (8 bits):",
                value="00000000",
                max_chars=8,
                key="index_binary",
            )
            try:
                if len(binary_input) == 8 and all(c in "01" for c in binary_input):
                    matrix_idx = int(binary_input, 2)
                else:
                    st.error("Please enter exactly 8 binary digits")
                    matrix_idx = 0
            except ValueError:
                st.error("Invalid binary input. Please enter only 0s and 1s.")
                matrix_idx = 0

        if st.button("🔍 View Matrix", key="view_matrix_btn"):
            matrix = explorer.index_to_matrix(matrix_idx)
            first_row = matrix[0]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Matrix:**")
                matrix_df = pd.DataFrame(
                    matrix,
                    columns=[f"Col {i}" for i in range(8)],
                    index=[f"Row {i}" for i in range(8)],
                )
                st.dataframe(matrix_df, width="stretch")

            with col2:
                st.write("**Properties:**")
                st.metric("Decimal Index", matrix_idx)
                st.metric("Binary Index", format(matrix_idx, "08b"))
                st.metric("First Row", "".join(map(str, first_row)))
                st.metric("Rank", np.linalg.matrix_rank(matrix))
                st.metric("Ones Count", int(np.sum(matrix)))

    # Tab 4: Range Explorer
    with tab4:
        st.subheader("Explore Matrix Ranges")

        st.write("View a sequence of consecutive matrices:")

        col1, col2 = st.columns(2)

        with col1:
            start_index = st.number_input(
                "Start Index:",
                min_value=0,
                max_value=245,
                value=0,
                step=1,
                key="range_start",
            )

        with col2:
            count = st.slider(
                "Number of matrices:",
                min_value=1,
                max_value=10,
                value=5,
                key="range_count",
            )

        if st.button("📊 Generate Range", key="generate_range_btn"):
            matrices = explorer.get_matrix_range(start_index, count)

            for i, matrix in enumerate(matrices):
                current_idx = start_index + i
                first_row = matrix[0]

                with st.expander(
                    f"Matrix {current_idx} (First Row: {''.join(map(str, first_row))})",
                    expanded=(i == 0),
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        matrix_df = pd.DataFrame(
                            matrix,
                            columns=[f"Col {i}" for i in range(8)],
                            index=[f"Row {i}" for i in range(8)],
                        )
                        st.dataframe(matrix_df, width="stretch")

                    with col2:
                        st.metric("Index", current_idx)
                        st.metric("First Row", "".join(map(str, first_row)))
                        st.metric("Rank", np.linalg.matrix_rank(matrix))
                        st.metric("Ones", int(np.sum(matrix)))

    # Information footer
    with st.expander("ℹ️ About Affine Matrix Structure"):
        st.markdown(
            """
        ### Circular Shift Affine Matrix
        
        The affine matrices used in this implementation follow a special structure:
        
        1. **First Row:** You specify an 8-bit binary pattern
        2. **Remaining Rows:** Each row is generated by circular right shift of the previous row
        
        #### Example:
        
        If the first row is `[0, 0, 0, 0, 0, 1, 1, 1]`, then:
        ```
        Row 0: [0, 0, 0, 0, 0, 1, 1, 1]  ← First row (input)
        Row 1: [1, 0, 0, 0, 0, 0, 1, 1]  ← Shift right by 1
        Row 2: [1, 1, 0, 0, 0, 0, 0, 1]  ← Shift right by 1
        Row 3: [1, 1, 1, 0, 0, 0, 0, 0]  ← Shift right by 1
        Row 4: [0, 1, 1, 1, 0, 0, 0, 0]  ← Shift right by 1
        Row 5: [0, 0, 1, 1, 1, 0, 0, 0]  ← Shift right by 1
        Row 6: [0, 0, 0, 1, 1, 1, 0, 0]  ← Shift right by 1
        Row 7: [0, 0, 0, 0, 1, 1, 1, 1]  ← Shift right by 1
        ```
        
        ### Total Possibilities
        
        - Since only the first row needs to be specified (8 bits)
        - Total possible matrices: **2^8 = 256**
        - Much more manageable than 2^64 full exploration!
        
        ### Matrix Properties in GF(2)
        
        Each element is in GF(2) (Galois Field with 2 elements: 0 or 1).
        Operations are performed modulo 2.
        """
        )


# Import pandas at the top
import pandas as pd


# Main execution for standalone testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Affine Matrix Explorer", page_icon="🔍", layout="wide"
    )

    render_affine_matrix_explorer()
