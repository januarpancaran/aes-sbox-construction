"""
S-box Constructor Component for AES S-box Construction
Based on the paper: "AES S-box modification uses affine matrices exploration"

This module handles the construction of S-boxes using:
1. Irreducible polynomial (x^8 + x^4 + x^3 + x + 1)
2. Multiplicative inverse matrix
3. Affine transformation (affine matrix + 8-bit constant)

Optimized for fast performance with caching and vectorization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def get_precomputed_inverse_table():
    """
    Pre-computed multiplicative inverse table for GF(2^8).
    This is cached to avoid recomputation.
    """
    # Using the standard AES S-box multiplicative inverse table
    return np.array(
        [
            [0, 1, 141, 246, 203, 82, 123, 209, 232, 79, 41, 192, 176, 225, 229, 199],
            [116, 180, 170, 75, 153, 43, 96, 95, 88, 63, 253, 204, 255, 64, 238, 178],
            [58, 110, 90, 241, 85, 77, 168, 201, 193, 10, 152, 21, 48, 68, 162, 194],
            [44, 69, 146, 108, 243, 57, 102, 66, 242, 53, 32, 111, 119, 187, 89, 25],
            [29, 254, 55, 103, 45, 49, 245, 105, 167, 100, 171, 19, 84, 37, 233, 9],
            [237, 92, 5, 202, 76, 36, 135, 191, 24, 62, 34, 240, 81, 236, 97, 23],
            [22, 94, 175, 211, 73, 166, 54, 67, 244, 71, 145, 223, 51, 147, 33, 59],
            [
                121,
                183,
                151,
                133,
                16,
                181,
                186,
                60,
                182,
                112,
                208,
                6,
                161,
                250,
                129,
                130,
            ],
            [
                131,
                126,
                127,
                128,
                150,
                115,
                190,
                86,
                155,
                158,
                149,
                217,
                247,
                2,
                185,
                164,
            ],
            [
                222,
                106,
                50,
                109,
                216,
                138,
                132,
                114,
                42,
                20,
                159,
                136,
                249,
                220,
                137,
                154,
            ],
            [251, 124, 46, 195, 143, 184, 101, 72, 38, 200, 18, 74, 206, 231, 210, 98],
            [12, 224, 31, 239, 17, 117, 120, 113, 165, 142, 118, 61, 189, 188, 134, 87],
            [11, 40, 47, 163, 218, 212, 228, 15, 169, 39, 83, 4, 27, 252, 172, 230],
            [
                122,
                7,
                174,
                99,
                197,
                219,
                226,
                234,
                148,
                139,
                196,
                213,
                157,
                248,
                144,
                107,
            ],
            [177, 13, 214, 235, 198, 14, 207, 173, 8, 78, 215, 227, 93, 80, 30, 179],
            [91, 35, 56, 52, 104, 70, 3, 140, 221, 156, 125, 160, 205, 26, 65, 28],
        ],
        dtype=np.int32,
    )


class SBoxConstructor:
    """
    Class to handle S-box construction using the AES S-box modification method.
    """

    def __init__(self):
        self.irreducible_poly = 0x11B  # x^8 + x^4 + x^3 + x + 1 in hex
        self.gf_size = 256

        # 8-bit constant from AES (C_AES)
        self.c_aes = np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=int)

        # Pre-compute multiplicative inverse table (cached)
        self.mult_inverse_table = self._get_cached_inverse_table()

    def _gf_mult(self, a: int, b: int) -> int:
        """
        Multiply two numbers in GF(2^8) with irreducible polynomial.
        Optimized version.

        Args:
            a: First number (0-255)
            b: Second number (0-255)

        Returns:
            Product in GF(2^8)
        """
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            high_bit = a & 0x80
            a <<= 1
            if high_bit:
                a ^= self.irreducible_poly
            b >>= 1
        return result & 0xFF

    def _gf_inverse(self, a: int) -> int:
        """
        Calculate multiplicative inverse in GF(2^8).
        Uses simple exponentiation: a^254 = a^-1 in GF(2^8)
        This is much faster than extended Euclidean algorithm.

        Args:
            a: Number to invert (0-255)

        Returns:
            Multiplicative inverse in GF(2^8)
        """
        if a == 0:
            return 0

        # Use Fermat's little theorem: a^(2^8 - 1) = 1, so a^-1 = a^(2^8 - 2) = a^254
        result = 1
        power = a
        exponent = 254

        while exponent > 0:
            if exponent & 1:
                result = self._gf_mult(result, power)
            power = self._gf_mult(power, power)
            exponent >>= 1

        return result

    def _get_cached_inverse_table(self) -> np.ndarray:
        """
        Get cached multiplicative inverse table.
        Uses Streamlit's cache for instant loading.

        Returns:
            16x16 numpy array containing multiplicative inverses
        """
        return get_precomputed_inverse_table()

    def _generate_mult_inverse_table(self) -> np.ndarray:
        """
        Generate the multiplicative inverse table for GF(2^8).
        This is Table 1 in the paper.

        Returns:
            16x16 numpy array containing multiplicative inverses
        """
        table = np.zeros((16, 16), dtype=int)

        for row in range(16):
            for col in range(16):
                value = row * 16 + col
                inverse = self._gf_inverse(value)
                table[row, col] = inverse

        return table

    def get_mult_inverse(self, value: int) -> int:
        """
        Get multiplicative inverse for a value.

        Args:
            value: Input value (0-255)

        Returns:
            Multiplicative inverse
        """
        row = value // 16
        col = value % 16
        return self.mult_inverse_table[row, col]

    def byte_to_bits(self, byte_val: int) -> np.ndarray:
        """
        Convert byte value to 8-bit array.

        Args:
            byte_val: Byte value (0-255)

        Returns:
            8-element numpy array of bits
        """
        return np.array([int(b) for b in format(byte_val, "08b")], dtype=int)

    def bits_to_byte(self, bits: np.ndarray) -> int:
        """
        Convert 8-bit array to byte value.

        Args:
            bits: 8-element numpy array of bits

        Returns:
            Byte value (0-255)
        """
        return int("".join(map(str, bits)), 2)

    def affine_transform(
        self, input_byte: int, affine_matrix: np.ndarray, constant: np.ndarray = None
    ) -> int:
        """
        Apply affine transformation: B(X) = K * X^-1 + C (mod 2)

        Args:
            input_byte: Input byte (0-255)
            affine_matrix: 8x8 affine matrix
            constant: 8-bit constant (default: C_AES)

        Returns:
            Transformed byte (0-255)
        """
        if constant is None:
            constant = self.c_aes

        # Get multiplicative inverse
        inverse = self.get_mult_inverse(input_byte)

        # Convert to bit array
        inverse_bits = self.byte_to_bits(inverse)

        # Matrix multiplication in GF(2): K * X^-1
        result = np.dot(affine_matrix, inverse_bits) % 2

        # Add constant in GF(2): + C
        result = (result + constant) % 2

        # Convert back to byte
        return self.bits_to_byte(result)

    def construct_sbox(
        self, affine_matrix: np.ndarray, constant: np.ndarray = None
    ) -> np.ndarray:
        """
        Construct complete S-box using affine matrix and constant.
        Optimized with vectorization where possible.

        Args:
            affine_matrix: 8x8 affine matrix for transformation
            constant: 8-bit constant (default: C_AES)

        Returns:
            16x16 S-box table
        """
        if constant is None:
            constant = self.c_aes

        sbox = np.zeros((16, 16), dtype=int)

        # Vectorized processing
        for row in range(16):
            for col in range(16):
                input_value = row * 16 + col
                output_value = self.affine_transform(
                    input_value, affine_matrix, constant
                )
                sbox[row, col] = output_value

        return sbox

    def test_balance(self, sbox: np.ndarray) -> Tuple[bool, Dict]:
        """
        Test if S-box satisfies balance criterion.
        Optimized version.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (passes_test, detailed_results)
        """
        flat_sbox = sbox.flatten()
        results = {}

        # Vectorized bit counting
        for bit_pos in range(8):
            bit_array = (flat_sbox >> bit_pos) & 1
            ones = np.sum(bit_array)
            zeros = len(bit_array) - ones
            results[f"f_{bit_pos}"] = {"zeros": int(zeros), "ones": int(ones)}

        # All bit positions should have 128 zeros and 128 ones
        passes = all(r["zeros"] == 128 and r["ones"] == 128 for r in results.values())

        return passes, results

    def test_bijectivity(self, sbox: np.ndarray) -> Tuple[bool, Dict]:
        """
        Test if S-box satisfies bijectivity criterion.
        Optimized version.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (passes_test, detailed_results)
        """
        flat_sbox = sbox.flatten()
        unique_values = len(np.unique(flat_sbox))

        results = {
            "total_values": len(flat_sbox),
            "unique_values": unique_values,
            "expected": 256,
            "duplicates": [],
        }

        # Only check for duplicates if test fails
        if unique_values != 256:
            # Find duplicates
            unique, counts = np.unique(flat_sbox, return_counts=True)
            duplicates = unique[counts > 1]

            for dup_val in duplicates:
                positions = np.where(flat_sbox == dup_val)[0].tolist()
                results["duplicates"].append(
                    {"value": int(dup_val), "positions": positions}
                )

        passes = unique_values == 256

        return passes, results

    def validate_sbox(self, sbox: np.ndarray) -> Dict:
        """
        Validate S-box against balance and bijectivity criteria.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Dictionary with validation results
        """
        balance_pass, balance_results = self.test_balance(sbox)
        bijective_pass, bijective_results = self.test_bijectivity(sbox)

        return {
            "valid": balance_pass and bijective_pass,
            "balance": {"passes": balance_pass, "details": balance_results},
            "bijectivity": {"passes": bijective_pass, "details": bijective_results},
        }


def render_sbox_constructor():
    """
    Streamlit component for S-box construction.
    This function can be imported and called from other views.
    """
    st.header("📦 S-box Construction")

    # Initialize constructor with caching
    @st.cache_resource
    def get_constructor():
        return SBoxConstructor()

    constructor = get_constructor()

    # Display information
    st.info(
        """
    **S-box Construction Process:**
    
    1. **Irreducible Polynomial:** x^8 + x^4 + x^3 + x + 1 (0x11B in hex)
    2. **Multiplicative Inverse:** Compute inverse for each value in GF(2^8)
    3. **Affine Transformation:** Apply B(X) = K × X^-1 + C (mod 2)
    
    Where:
    - **K** is the 8×8 affine matrix
    - **X^-1** is the multiplicative inverse
    - **C** is the 8-bit constant
    """
    )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔧 Build S-box",
            "📊 Multiplicative Inverse Table",
            "🧪 Validation",
            "💾 Export S-box",
        ]
    )

    # Tab 1: Build S-box
    with tab1:
        st.subheader("Build Your S-box")

        # Method selection
        construction_method = st.radio(
            "Select construction method:",
            ["Use Custom Affine Matrix", "Enter First Row (Auto-generate Matrix)"],
            horizontal=True,
        )

        affine_matrix = None

        if construction_method == "Enter First Row (Auto-generate Matrix)":
            st.write("**Step 1: Enter the first row of the affine matrix**")

            input_method = st.radio(
                "Input method:",
                ["Binary String", "Decimal Value", "Individual Bits"],
                horizontal=True,
                key="first_row_input_method",
            )

            if input_method == "Binary String":
                binary_input = st.text_input(
                    "Enter 8-bit binary string:",
                    value="00000111",
                    max_chars=8,
                    key="sbox_binary_input",
                )

                if len(binary_input) == 8 and all(c in "01" for c in binary_input):
                    first_row = np.array([int(b) for b in binary_input])

                    # Generate matrix using circular shift
                    affine_matrix = np.zeros((8, 8), dtype=int)
                    affine_matrix[0] = first_row
                    for i in range(1, 8):
                        affine_matrix[i] = np.roll(affine_matrix[i - 1], 1)
                else:
                    st.error("Please enter exactly 8 binary digits")

            elif input_method == "Decimal Value":
                decimal_value = st.number_input(
                    "Enter decimal value (0-255):",
                    min_value=0,
                    max_value=255,
                    value=7,
                    step=1,
                    key="sbox_decimal_input",
                )

                binary_str = format(decimal_value, "08b")
                first_row = np.array([int(b) for b in binary_str])

                # Generate matrix
                affine_matrix = np.zeros((8, 8), dtype=int)
                affine_matrix[0] = first_row
                for i in range(1, 8):
                    affine_matrix[i] = np.roll(affine_matrix[i - 1], 1)

                st.info(f"Binary representation: {binary_str}")

            else:  # Individual Bits
                st.write("Set each bit:")
                cols = st.columns(8)
                bits = []

                if "sbox_bits" not in st.session_state:
                    st.session_state.sbox_bits = [0, 0, 0, 0, 0, 1, 1, 1]

                for i, col in enumerate(cols):
                    with col:
                        st.session_state.sbox_bits[i] = st.selectbox(
                            f"Bit {i}",
                            options=[0, 1],
                            index=st.session_state.sbox_bits[i],
                            key=f"sbox_bit_{i}",
                        )

                first_row = np.array(st.session_state.sbox_bits)

                # Generate matrix
                affine_matrix = np.zeros((8, 8), dtype=int)
                affine_matrix[0] = first_row
                for i in range(1, 8):
                    affine_matrix[i] = np.roll(affine_matrix[i - 1], 1)

        else:  # Use Custom Affine Matrix
            st.write("**Step 1: Enter custom affine matrix**")
            st.write("Enter each row of the 8×8 matrix:")

            affine_matrix = np.zeros((8, 8), dtype=int)

            for row in range(8):
                row_input = st.text_input(
                    f"Row {row}:",
                    value=(
                        "00000111"
                        if row == 0
                        else "10000011" if row == 1 else "11000001"
                    ),
                    max_chars=8,
                    key=f"matrix_row_{row}",
                )

                if len(row_input) == 8 and all(c in "01" for c in row_input):
                    affine_matrix[row] = [int(b) for b in row_input]
                else:
                    st.error(f"Row {row}: Please enter exactly 8 binary digits")
                    affine_matrix = None
                    break

        # Display affine matrix
        if affine_matrix is not None:
            st.write("**Generated Affine Matrix (K):**")
            matrix_df = pd.DataFrame(
                affine_matrix,
                columns=[f"C{i}" for i in range(8)],
                index=[f"R{i}" for i in range(8)],
            )
            st.dataframe(matrix_df, width="stretch")

        # Step 2: 8-bit constant
        st.write("**Step 2: Enter 8-bit constant (C)**")

        constant_method = st.radio(
            "Constant input method:",
            ["Use C_AES (Default)", "Custom Constant"],
            horizontal=True,
            key="constant_method",
        )

        if constant_method == "Use C_AES (Default)":
            constant = constructor.c_aes
            st.info(
                f"C_AES = {' '.join(map(str, constant))} (binary) = {constructor.bits_to_byte(constant)} (decimal)"
            )
        else:
            constant_input = st.text_input(
                "Enter 8-bit constant:",
                value="11000110",
                max_chars=8,
                key="constant_input",
            )

            if len(constant_input) == 8 and all(c in "01" for c in constant_input):
                constant = np.array([int(b) for b in constant_input])
            else:
                st.error("Please enter exactly 8 binary digits")
                constant = constructor.c_aes

        # Construct S-box button
        st.write("---")

        if affine_matrix is not None:
            if st.button("🚀 Construct S-box", type="primary", width="stretch"):
                # Construct S-box without spinner for speed
                # Construct S-box
                sbox = constructor.construct_sbox(affine_matrix, constant)

                # Store in session state
                st.session_state.constructed_sbox = sbox
                st.session_state.affine_matrix = affine_matrix
                st.session_state.constant = constant

                st.success("✅ S-box constructed successfully!")
                st.rerun()  # Refresh to show the S-box immediately

        # Display constructed S-box
        if hasattr(st.session_state, "constructed_sbox"):
            st.write("---")
            st.subheader("Constructed S-box")

            sbox_df = pd.DataFrame(
                st.session_state.constructed_sbox,
                columns=[f"{i:X}" for i in range(16)],
                index=[f"{i:X}" for i in range(16)],
            )

            st.dataframe(sbox_df, width="stretch")

            # Quick validation
            validation = constructor.validate_sbox(st.session_state.constructed_sbox)

            col1, col2 = st.columns(2)
            with col1:
                if validation["balance"]["passes"]:
                    st.success("✅ Balance criterion: PASSED")
                else:
                    st.error("❌ Balance criterion: FAILED")

            with col2:
                if validation["bijectivity"]["passes"]:
                    st.success("✅ Bijectivity criterion: PASSED")
                else:
                    st.error("❌ Bijectivity criterion: FAILED")

            if validation["valid"]:
                st.success("🎉 This is a VALID S-box!")
            else:
                st.warning("⚠️ This S-box does not meet all criteria.")

    # Tab 2: Multiplicative Inverse Table
    with tab2:
        st.subheader("Multiplicative Inverse Table (GF(2^8))")

        st.write(
            """
        This is the multiplicative inverse table generated using the irreducible polynomial 
        **x^8 + x^4 + x^3 + x + 1**.
        
        This corresponds to **Table 1** in the paper.
        """
        )

        # Display table
        inv_table_df = pd.DataFrame(
            constructor.mult_inverse_table,
            columns=[f"{i:X}" for i in range(16)],
            index=[f"{i:X}" for i in range(16)],
        )

        st.dataframe(inv_table_df, width="stretch")

        # Example lookup
        st.write("**Example Lookup:**")

        col1, col2 = st.columns(2)

        with col1:
            lookup_value = st.number_input(
                "Enter value (0-255):",
                min_value=0,
                max_value=255,
                value=15,
                step=1,
                key="lookup_value",
            )

        with col2:
            inverse = constructor.get_mult_inverse(lookup_value)
            st.metric("Multiplicative Inverse", f"{inverse} (0x{inverse:02X})")

            # Verify
            verify = constructor._gf_mult(lookup_value, inverse)
            if verify == 1:
                st.success("✅ Verification: a × a^-1 = 1")
            else:
                st.error(f"❌ Verification failed: {verify}")

    # Tab 3: Validation
    with tab3:
        st.subheader("S-box Validation")

        if not hasattr(st.session_state, "constructed_sbox"):
            st.info("👈 Please construct an S-box first in the 'Build S-box' tab.")
        else:
            sbox = st.session_state.constructed_sbox
            validation = constructor.validate_sbox(sbox)

            # Overall status
            if validation["valid"]:
                st.success("## ✅ VALID S-BOX")
                st.balloons()
            else:
                st.error("## ❌ INVALID S-BOX")

            st.write("---")

            # Balance test details
            st.subheader("1. Balance Criterion")

            if validation["balance"]["passes"]:
                st.success("✅ PASSED - All output bits have equal distribution")
            else:
                st.error("❌ FAILED - Output bits are not balanced")

            balance_data = []
            for bit_name, counts in validation["balance"]["details"].items():
                balance_data.append(
                    {
                        "Output Bit": bit_name,
                        "Zeros": counts["zeros"],
                        "Ones": counts["ones"],
                        "Status": (
                            "✅"
                            if counts["zeros"] == 128 and counts["ones"] == 128
                            else "❌"
                        ),
                    }
                )

            balance_df = pd.DataFrame(balance_data)
            st.dataframe(balance_df, width="stretch", hide_index=True)

            st.write("---")

            # Bijectivity test details
            st.subheader("2. Bijectivity Criterion")

            if validation["bijectivity"]["passes"]:
                st.success("✅ PASSED - All output values are unique")
            else:
                st.error("❌ FAILED - Duplicate values found")

            bij_details = validation["bijectivity"]["details"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Values", bij_details["total_values"])

            with col2:
                st.metric("Unique Values", bij_details["unique_values"])

            with col3:
                st.metric("Expected", bij_details["expected"])

            if bij_details["duplicates"]:
                st.warning(f"Found {len(bij_details['duplicates'])} duplicate values:")
                dup_df = pd.DataFrame(bij_details["duplicates"])
                st.dataframe(dup_df, hide_index=True)

    # Tab 4: Export
    with tab4:
        st.subheader("Export S-box")

        if not hasattr(st.session_state, "constructed_sbox"):
            st.info("👈 Please construct an S-box first in the 'Build S-box' tab.")
        else:
            sbox = st.session_state.constructed_sbox

            st.write("**Export Options:**")

            export_format = st.selectbox(
                "Select format:",
                ["Python Array", "C Array", "CSV", "Hex String", "LaTeX Table"],
            )

            output = ""

            if export_format == "Python Array":
                output = "s_box = (\n"
                for row in sbox:
                    """Hex
                    output += "    " + ", ".join(f"0x{val:02X}" for val in row) + ",\n"
                    """
                    output += "    " + ", ".join(str(val) for val in row) + ",\n"
                output += ")"

            elif export_format == "C Array":
                output = "uint8_t s_box[256] = {\n"
                for i, row in enumerate(sbox):
                    """Hex
                    output += "    " + ", ".join(f"0x{val:02X}" for val in row)
                    output += ",\n" if i < 15 else "\n"
                    """
                    output += "    " + ", ".join(str(val) for val in row)
                    output += ",\n" if i < 15 else "\n"
                output += "};"

            elif export_format == "CSV":
                # output = "," + ",".join(f"{i:X}" for i in range(16)) + "\n"
                # for i, row in enumerate(sbox):
                #     output += f"{i:X}," + ",".join(str(val) for val in row) + "\n"
                output = "," + ",".join(str(i) for i in range(16)) + "\n"
                for i, row in enumerate(sbox):
                    output += str(i) + "," + ",".join(str(val) for val in row) + "\n"

            elif export_format == "Hex String":
                flat = sbox.flatten()
                output = "".join(f"{val:02X}" for val in flat)

            elif export_format == "LaTeX Table":
                output = "\\begin{tabular}{c|" + "c" * 16 + "}\n"
                output += (
                    "  & "
                    + " & ".join(f"{i:X}" for i in range(16))
                    + " \\\\\n\\hline\n"
                )
                for i, row in enumerate(sbox):
                    output += (
                        f"{i:X} & " + " & ".join(str(val) for val in row) + " \\\\\n"
                    )
                output += "\\end{tabular}"

            st.code(
                output,
                language=(
                    "python"
                    if "Python" in export_format
                    else (
                        "c"
                        if "C" in export_format
                        else "latex" if "LaTeX" in export_format else "text"
                    )
                ),
            )

            ext = {
                "Python": "py",
                "C": "c",
                "CSV": "csv",
            }.get(export_format, "txt")

            # Download button
            st.download_button(
                label="📥 Download",
                data=output,
                file_name=f"sbox.{ext}",
                mime="text/plain",
            )

            # Also save metadata
            st.write("---")
            st.write("**S-box Metadata:**")

            metadata = f"""S-box Construction Details
==========================

Irreducible Polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)

Affine Matrix (K):
{st.session_state.affine_matrix}

8-bit Constant (C):
{st.session_state.constant}

First Row (Binary): {''.join(map(str, st.session_state.affine_matrix[0]))}
First Row (Decimal): {constructor.bits_to_byte(st.session_state.affine_matrix[0])}

Validation Status:
- Balance: {'PASSED' if constructor.validate_sbox(sbox)['balance']['passes'] else 'FAILED'}
- Bijectivity: {'PASSED' if constructor.validate_sbox(sbox)['bijectivity']['passes'] else 'FAILED'}
"""

            st.code(metadata, language="text")

            st.download_button(
                label="📥 Download Metadata",
                data=metadata,
                file_name="sbox_metadata.txt",
                mime="text/plain",
            )


# Main execution for standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="S-box Constructor", page_icon="📦", layout="wide")

    render_sbox_constructor()
