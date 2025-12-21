"""
S-box Testing Component for AES S-box Construction
Based on the paper: "AES S-box modification uses affine matrices exploration"

This module implements cryptographic strength tests:
1. Nonlinearity (NL)
2. Strict Avalanche Criterion (SAC)
3. Bit Independence Criterion - Nonlinearity (BIC-NL)
4. Bit Independence Criterion - Strict Avalanche Criterion (BIC-SAC)
5. Linear Approximation Probability (LAP)
6. Differential Approximation Probability (DAP)
7. Differential Uniformity (DU)
8. Algebraic Degree (AD)
9. Transparency Order (TO)
10. Correlation Immunity (CI)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


class SBoxTester:
    """
    Class to handle cryptographic strength testing of S-boxes.
    """

    def __init__(self):
        self.n = 8  # 8-bit S-box
        self.size = 2**self.n  # 256 elements

    def hamming_distance(self, a: int, b: int) -> int:
        """
        Calculate Hamming distance between two integers.

        Args:
            a, b: Two integers

        Returns:
            Hamming distance (number of differing bits)
        """
        xor = a ^ b
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance

    def walsh_hadamard_transform(self, truth_table: np.ndarray) -> np.ndarray:
        """
        Compute Walsh-Hadamard Transform of a Boolean function.

        Args:
            truth_table: Truth table of the Boolean function

        Returns:
            Walsh-Hadamard spectrum
        """
        n = len(truth_table)
        # Convert to bipolar form: 0 -> 1, 1 -> -1
        bipolar = 1 - 2 * truth_table

        # Fast Walsh-Hadamard Transform
        h = bipolar.copy()
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u = h[i + j]
                    v = h[i + j + step]
                    h[i + j] = u + v
                    h[i + j + step] = u - v
            step *= 2

        return h

    def test_nonlinearity(self, sbox: np.ndarray) -> Tuple[int, Dict]:
        """
        Test Nonlinearity (NL) of S-box.
        NL measures resistance to linear cryptanalysis.
        Ideal value: 112 for 8-bit S-boxes.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (NL value, details)
        """
        flat_sbox = sbox.flatten()
        nl_values = []

        # For each output bit
        for bit_pos in range(8):
            # Extract truth table for this bit
            truth_table = np.array([(flat_sbox[x] >> bit_pos) & 1 for x in range(256)])

            # Compute Walsh-Hadamard spectrum
            spectrum = self.walsh_hadamard_transform(truth_table)

            # Nonlinearity = 2^(n-1) - max|W(f)|/2
            max_walsh = np.max(np.abs(spectrum))
            nl = 2 ** (self.n - 1) - max_walsh // 2
            nl_values.append(nl)

        # Overall NL is the minimum across all output bits
        overall_nl = int(np.min(nl_values))

        details = {
            "overall": overall_nl,
            "per_bit": {f"f_{i}": int(nl_values[i]) for i in range(8)},
            "min": int(np.min(nl_values)),
            "max": int(np.max(nl_values)),
            "avg": float(np.mean(nl_values)),
            "ideal": 112,
        }

        return overall_nl, details

    def test_sac(self, sbox: np.ndarray) -> Tuple[float, Dict]:
        """
        Test Strict Avalanche Criterion (SAC).
        Measures avalanche effect - how output changes when input changes by 1 bit.
        Ideal value: 0.5

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (SAC value, details)
        """
        flat_sbox = sbox.flatten()
        sac_matrix = np.zeros((8, 8))  # [input_bit][output_bit]

        # For each input bit position
        for input_bit in range(8):
            # For each input value
            for x in range(256):
                # Flip the input bit
                x_prime = x ^ (1 << input_bit)

                # Get outputs
                y = flat_sbox[x]
                y_prime = flat_sbox[x_prime]

                # Count changed output bits
                for output_bit in range(8):
                    if ((y >> output_bit) & 1) != ((y_prime >> output_bit) & 1):
                        sac_matrix[input_bit][output_bit] += 1

        # Normalize by 256 (total inputs)
        sac_matrix = sac_matrix / 256.0

        # Overall SAC is the average
        overall_sac = float(np.mean(sac_matrix))

        details = {
            "overall": overall_sac,
            "matrix": sac_matrix,
            "min": float(np.min(sac_matrix)),
            "max": float(np.max(sac_matrix)),
            "std": float(np.std(sac_matrix)),
            "ideal": 0.5,
            "deviation": abs(overall_sac - 0.5),
        }

        return overall_sac, details

    def test_bic_nl(self, sbox: np.ndarray) -> Tuple[int, Dict]:
        """
        Test Bit Independence Criterion - Nonlinearity (BIC-NL).
        Measures independence between different output bits.
        Ideal value: 112 for 8-bit S-boxes.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (BIC-NL value, details)
        """
        flat_sbox = sbox.flatten()
        bic_nl_matrix = np.zeros((8, 8))

        # For each pair of output bits
        for i in range(8):
            for j in range(i + 1, 8):
                # Create combined truth table (XOR of two bits)
                truth_table = np.array(
                    [
                        ((flat_sbox[x] >> i) & 1) ^ ((flat_sbox[x] >> j) & 1)
                        for x in range(256)
                    ]
                )

                # Compute Walsh-Hadamard spectrum
                spectrum = self.walsh_hadamard_transform(truth_table)

                # Nonlinearity
                max_walsh = np.max(np.abs(spectrum))
                nl = 2 ** (self.n - 1) - max_walsh // 2

                bic_nl_matrix[i][j] = nl
                bic_nl_matrix[j][i] = nl  # Symmetric

        # Overall BIC-NL is the minimum of off-diagonal elements
        mask = ~np.eye(8, dtype=bool)
        overall_bic_nl = int(np.min(bic_nl_matrix[mask]))

        details = {
            "overall": overall_bic_nl,
            "matrix": bic_nl_matrix,
            "min": int(np.min(bic_nl_matrix[mask])),
            "max": int(np.max(bic_nl_matrix[mask])),
            "avg": float(np.mean(bic_nl_matrix[mask])),
            "ideal": 112,
        }

        return overall_bic_nl, details

    def test_bic_sac(self, sbox: np.ndarray) -> Tuple[float, Dict]:
        """
        Test Bit Independence Criterion - Strict Avalanche Criterion (BIC-SAC).
        Measures independence of avalanche effect across output bits.
        Ideal value: 0.5

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (BIC-SAC value, details)
        """
        flat_sbox = sbox.flatten()
        bic_sac_matrix = np.zeros((8, 8))

        # For each pair of output bits
        for i in range(8):
            for j in range(i + 1, 8):
                count = 0

                # For each input
                for x in range(256):
                    for input_bit in range(8):
                        # Flip input bit
                        x_prime = x ^ (1 << input_bit)

                        y = flat_sbox[x]
                        y_prime = flat_sbox[x_prime]

                        # Check if both bits changed
                        bit_i_changed = ((y >> i) & 1) != ((y_prime >> i) & 1)
                        bit_j_changed = ((y >> j) & 1) != ((y_prime >> j) & 1)

                        # XOR: both changed or neither changed = 0, one changed = 1
                        if bit_i_changed != bit_j_changed:
                            count += 1

                # Normalize
                bic_sac_value = count / (256 * 8)
                bic_sac_matrix[i][j] = bic_sac_value
                bic_sac_matrix[j][i] = bic_sac_value

        # Overall is average of off-diagonal elements
        mask = ~np.eye(8, dtype=bool)
        overall_bic_sac = float(np.mean(bic_sac_matrix[mask]))

        details = {
            "overall": overall_bic_sac,
            "matrix": bic_sac_matrix,
            "min": float(np.min(bic_sac_matrix[mask])),
            "max": float(np.max(bic_sac_matrix[mask])),
            "std": float(np.std(bic_sac_matrix[mask])),
            "ideal": 0.5,
            "deviation": abs(overall_bic_sac - 0.5),
        }

        return overall_bic_sac, details

    def test_lap(self, sbox: np.ndarray) -> Tuple[float, Dict]:
        """
        Test Linear Approximation Probability (LAP).
        Measures resistance to linear cryptanalysis.
        Ideal value: 0.0625 (1/16) for 8-bit S-boxes.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (LAP value, details)
        """
        flat_sbox = sbox.flatten()
        max_bias = 0
        best_masks = {"input": 0, "output": 0}

        # For all non-zero input and output masks
        for input_mask in range(1, 256):
            for output_mask in range(1, 256):
                count = 0

                # Count matches
                for x in range(256):
                    # Compute input parity
                    input_parity = bin(x & input_mask).count("1") % 2

                    # Compute output parity
                    output_parity = bin(flat_sbox[x] & output_mask).count("1") % 2

                    if input_parity == output_parity:
                        count += 1

                # Calculate bias
                bias = abs(count - 128) / 256.0

                if bias > max_bias:
                    max_bias = bias
                    best_masks["input"] = input_mask
                    best_masks["output"] = output_mask

        lap = max_bias

        details = {
            "lap": lap,
            "max_bias": max_bias,
            "best_input_mask": f"0x{best_masks['input']:02X}",
            "best_output_mask": f"0x{best_masks['output']:02X}",
            "ideal": 0.0625,
        }

        return lap, details

    def test_dap(self, sbox: np.ndarray) -> Tuple[float, Dict]:
        """
        Test Differential Approximation Probability (DAP).
        Measures resistance to differential cryptanalysis.
        Ideal value: 0.015625 (1/64) for 8-bit S-boxes.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (DAP value, details)
        """
        flat_sbox = sbox.flatten()
        max_count = 0
        best_deltas = {"input": 0, "output": 0}

        # Difference distribution table
        ddt = np.zeros((256, 256), dtype=int)

        # For all input pairs
        for x1 in range(256):
            for x2 in range(256):
                input_diff = x1 ^ x2
                output_diff = flat_sbox[x1] ^ flat_sbox[x2]
                ddt[input_diff][output_diff] += 1

        # Find maximum (excluding input diff = 0)
        for input_diff in range(1, 256):
            for output_diff in range(256):
                if ddt[input_diff][output_diff] > max_count:
                    max_count = ddt[input_diff][output_diff]
                    best_deltas["input"] = input_diff
                    best_deltas["output"] = output_diff

        dap = max_count / 256.0

        details = {
            "dap": dap,
            "max_count": int(max_count),
            "best_input_diff": f"0x{best_deltas['input']:02X}",
            "best_output_diff": f"0x{best_deltas['output']:02X}",
            "ddt": ddt,
            "ideal": 0.015625,
        }

        return dap, details

    def test_differential_uniformity(self, sbox: np.ndarray) -> Tuple[int, Dict]:
        """
        Test Differential Uniformity (DU).
        Measures the maximum value in the Difference Distribution Table (DDT).
        Lower values indicate better resistance to differential attacks.
        Ideal value: 4 for AES-like S-boxes.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (DU value, details)
        """
        flat_sbox = sbox.flatten()

        # Difference distribution table
        ddt = np.zeros((256, 256), dtype=int)

        # Build DDT
        for x in range(256):
            for delta_in in range(256):
                x_prime = x ^ delta_in
                delta_out = flat_sbox[x] ^ flat_sbox[x_prime]
                ddt[delta_in][delta_out] += 1

        # Find maximum (excluding delta_in = 0)
        max_du = 0
        best_deltas = {"input": 0, "output": 0}

        for delta_in in range(1, 256):
            for delta_out in range(256):
                if ddt[delta_in][delta_out] > max_du:
                    max_du = ddt[delta_in][delta_out]
                    best_deltas["input"] = delta_in
                    best_deltas["output"] = delta_out

        details = {
            "du": int(max_du),
            "best_input_diff": f"0x{best_deltas['input']:02X}",
            "best_output_diff": f"0x{best_deltas['output']:02X}",
            "ddt": ddt,
            "ideal": 4,
        }

        return int(max_du), details

    def test_algebraic_degree(self, sbox: np.ndarray) -> Tuple[int, Dict]:
        """
        Test Algebraic Degree (AD).
        Measures the degree of polynomial representation of the S-box.
        Higher values indicate better resistance to algebraic attacks.
        Ideal value: 7 for 8-bit S-boxes (maximum is n-1).

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (minimum AD across output bits, details)
        """
        flat_sbox = sbox.flatten()
        degrees = []

        # For each output bit
        for bit_pos in range(8):
            # Extract truth table for this output bit
            truth_table = np.array([(flat_sbox[x] >> bit_pos) & 1 for x in range(256)])

            # Compute Algebraic Normal Form (ANF) using Mobius transform
            anf = truth_table.copy()

            # Mobius transform
            for i in range(8):
                step = 1 << i
                for j in range(0, 256, step * 2):
                    for k in range(step):
                        anf[j + k + step] ^= anf[j + k]

            # Find maximum degree (highest bit position with coefficient 1)
            max_degree = 0
            for i in range(256):
                if anf[i] == 1:
                    # Count number of 1s in binary representation (Hamming weight)
                    degree = bin(i).count("1")
                    max_degree = max(max_degree, degree)

            degrees.append(max_degree)

        min_degree = int(np.min(degrees))

        details = {
            "min_degree": min_degree,
            "max_degree": int(np.max(degrees)),
            "avg_degree": float(np.mean(degrees)),
            "per_bit": {f"f_{i}": degrees[i] for i in range(8)},
            "ideal": 7,
        }

        return min_degree, details

    def test_transparency_order(self, sbox: np.ndarray) -> Tuple[float, Dict]:
        """
        Test Transparency Order (TO).
        Measures the average correlation between input and output bits.
        Lower values indicate better confusion property.
        Ideal value: close to 0.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (maximum TO, details)
        """
        flat_sbox = sbox.flatten()
        to_matrix = np.zeros((8, 8))  # [input_bit][output_bit]

        # For each input bit and output bit pair
        for i in range(8):
            for j in range(8):
                correlation_sum = 0

                # Calculate correlation
                for x in range(256):
                    input_bit = (x >> i) & 1
                    output_bit = (flat_sbox[x] >> j) & 1

                    # XOR gives 0 if same, 1 if different
                    # Convert to +1/-1 for correlation
                    correlation_sum += 1 if (input_bit == output_bit) else -1

                # Normalize to [-1, 1]
                correlation = abs(correlation_sum) / 256.0
                to_matrix[i][j] = correlation

        # Maximum transparency order
        max_to = float(np.max(to_matrix))

        details = {
            "max_to": max_to,
            "avg_to": float(np.mean(to_matrix)),
            "min_to": float(np.min(to_matrix)),
            "matrix": to_matrix,
            "ideal": 0.0,
        }

        return max_to, details

    def test_correlation_immunity(self, sbox: np.ndarray) -> Tuple[int, Dict]:
        """
        Test Correlation Immunity (CI).
        Measures resistance to correlation attacks.
        A function has CI of order m if its output is statistically independent
        of any m input variables.
        Higher values are better.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Tuple of (minimum CI order, details)
        """
        flat_sbox = sbox.flatten()
        ci_orders = []

        # For each output bit
        for bit_pos in range(8):
            # Extract truth table
            truth_table = np.array([(flat_sbox[x] >> bit_pos) & 1 for x in range(256)])

            # Compute Walsh-Hadamard spectrum
            spectrum = self.walsh_hadamard_transform(truth_table)

            # CI order is determined by the sparsity of Walsh spectrum
            # Check for each order m
            ci_order = 0

            for m in range(1, 8):
                # Check if all Walsh coefficients with Hamming weight <= m are zero
                is_ci_m = True

                for w in range(1, 256):
                    if bin(w).count("1") <= m:
                        if spectrum[w] != 0:
                            is_ci_m = False
                            break

                if is_ci_m:
                    ci_order = m
                else:
                    break

            ci_orders.append(ci_order)

        min_ci = int(np.min(ci_orders))

        details = {
            "min_ci": min_ci,
            "max_ci": int(np.max(ci_orders)),
            "avg_ci": float(np.mean(ci_orders)),
            "per_bit": {f"f_{i}": ci_orders[i] for i in range(8)},
            "ideal": 0,
        }

        return min_ci, details

    def comprehensive_test(self, sbox: np.ndarray) -> Dict:
        """
        Run all tests on the S-box.

        Args:
            sbox: 16x16 S-box table

        Returns:
            Dictionary with all test results
        """
        results = {}

        # Run all tests
        nl, nl_details = self.test_nonlinearity(sbox)
        sac, sac_details = self.test_sac(sbox)
        bic_nl, bic_nl_details = self.test_bic_nl(sbox)
        bic_sac, bic_sac_details = self.test_bic_sac(sbox)
        lap, lap_details = self.test_lap(sbox)
        dap, dap_details = self.test_dap(sbox)
        du, du_details = self.test_differential_uniformity(sbox)
        ad, ad_details = self.test_algebraic_degree(sbox)
        to, to_details = self.test_transparency_order(sbox)
        ci, ci_details = self.test_correlation_immunity(sbox)

        results = {
            "NL": {"value": nl, "details": nl_details},
            "SAC": {"value": sac, "details": sac_details},
            "BIC-NL": {"value": bic_nl, "details": bic_nl_details},
            "BIC-SAC": {"value": bic_sac, "details": bic_sac_details},
            "LAP": {"value": lap, "details": lap_details},
            "DAP": {"value": dap, "details": dap_details},
            "DU": {"value": du, "details": du_details},
            "AD": {"value": ad, "details": ad_details},
            "TO": {"value": to, "details": to_details},
            "CI": {"value": ci, "details": ci_details},
        }

        return results


def render_sbox_tester():
    """
    Streamlit component for S-box testing.
    This function can be imported and called from other views.
    """
    st.header("🧪 S-box Testing")

    # Initialize tester with caching
    @st.cache_resource
    def get_tester():
        return SBoxTester()

    tester = get_tester()

    # Check if S-box exists
    if not hasattr(st.session_state, "constructed_sbox"):
        st.warning("⚠️ No S-box found! Please construct an S-box first.")

        # Provide example S-box option
        if st.button("📥 Load Example S-box (AES)", width="stretch"):
            # Standard AES S-box
            aes_sbox = np.array(
                [
                    [
                        0x63,
                        0x7C,
                        0x77,
                        0x7B,
                        0xF2,
                        0x6B,
                        0x6F,
                        0xC5,
                        0x30,
                        0x01,
                        0x67,
                        0x2B,
                        0xFE,
                        0xD7,
                        0xAB,
                        0x76,
                    ],
                    [
                        0xCA,
                        0x82,
                        0xC9,
                        0x7D,
                        0xFA,
                        0x59,
                        0x47,
                        0xF0,
                        0xAD,
                        0xD4,
                        0xA2,
                        0xAF,
                        0x9C,
                        0xA4,
                        0x72,
                        0xC0,
                    ],
                    [
                        0xB7,
                        0xFD,
                        0x93,
                        0x26,
                        0x36,
                        0x3F,
                        0xF7,
                        0xCC,
                        0x34,
                        0xA5,
                        0xE5,
                        0xF1,
                        0x71,
                        0xD8,
                        0x31,
                        0x15,
                    ],
                    [
                        0x04,
                        0xC7,
                        0x23,
                        0xC3,
                        0x18,
                        0x96,
                        0x05,
                        0x9A,
                        0x07,
                        0x12,
                        0x80,
                        0xE2,
                        0xEB,
                        0x27,
                        0xB2,
                        0x75,
                    ],
                    [
                        0x09,
                        0x83,
                        0x2C,
                        0x1A,
                        0x1B,
                        0x6E,
                        0x5A,
                        0xA0,
                        0x52,
                        0x3B,
                        0xD6,
                        0xB3,
                        0x29,
                        0xE3,
                        0x2F,
                        0x84,
                    ],
                    [
                        0x53,
                        0xD1,
                        0x00,
                        0xED,
                        0x20,
                        0xFC,
                        0xB1,
                        0x5B,
                        0x6A,
                        0xCB,
                        0xBE,
                        0x39,
                        0x4A,
                        0x4C,
                        0x58,
                        0xCF,
                    ],
                    [
                        0xD0,
                        0xEF,
                        0xAA,
                        0xFB,
                        0x43,
                        0x4D,
                        0x33,
                        0x85,
                        0x45,
                        0xF9,
                        0x02,
                        0x7F,
                        0x50,
                        0x3C,
                        0x9F,
                        0xA8,
                    ],
                    [
                        0x51,
                        0xA3,
                        0x40,
                        0x8F,
                        0x92,
                        0x9D,
                        0x38,
                        0xF5,
                        0xBC,
                        0xB6,
                        0xDA,
                        0x21,
                        0x10,
                        0xFF,
                        0xF3,
                        0xD2,
                    ],
                    [
                        0xCD,
                        0x0C,
                        0x13,
                        0xEC,
                        0x5F,
                        0x97,
                        0x44,
                        0x17,
                        0xC4,
                        0xA7,
                        0x7E,
                        0x3D,
                        0x64,
                        0x5D,
                        0x19,
                        0x73,
                    ],
                    [
                        0x60,
                        0x81,
                        0x4F,
                        0xDC,
                        0x22,
                        0x2A,
                        0x90,
                        0x88,
                        0x46,
                        0xEE,
                        0xB8,
                        0x14,
                        0xDE,
                        0x5E,
                        0x0B,
                        0xDB,
                    ],
                    [
                        0xE0,
                        0x32,
                        0x3A,
                        0x0A,
                        0x49,
                        0x06,
                        0x24,
                        0x5C,
                        0xC2,
                        0xD3,
                        0xAC,
                        0x62,
                        0x91,
                        0x95,
                        0xE4,
                        0x79,
                    ],
                    [
                        0xE7,
                        0xC8,
                        0x37,
                        0x6D,
                        0x8D,
                        0xD5,
                        0x4E,
                        0xA9,
                        0x6C,
                        0x56,
                        0xF4,
                        0xEA,
                        0x65,
                        0x7A,
                        0xAE,
                        0x08,
                    ],
                    [
                        0xBA,
                        0x78,
                        0x25,
                        0x2E,
                        0x1C,
                        0xA6,
                        0xB4,
                        0xC6,
                        0xE8,
                        0xDD,
                        0x74,
                        0x1F,
                        0x4B,
                        0xBD,
                        0x8B,
                        0x8A,
                    ],
                    [
                        0x70,
                        0x3E,
                        0xB5,
                        0x66,
                        0x48,
                        0x03,
                        0xF6,
                        0x0E,
                        0x61,
                        0x35,
                        0x57,
                        0xB9,
                        0x86,
                        0xC1,
                        0x1D,
                        0x9E,
                    ],
                    [
                        0xE1,
                        0xF8,
                        0x98,
                        0x11,
                        0x69,
                        0xD9,
                        0x8E,
                        0x94,
                        0x9B,
                        0x1E,
                        0x87,
                        0xE9,
                        0xCE,
                        0x55,
                        0x28,
                        0xDF,
                    ],
                    [
                        0x8C,
                        0xA1,
                        0x89,
                        0x0D,
                        0xBF,
                        0xE6,
                        0x42,
                        0x68,
                        0x41,
                        0x99,
                        0x2D,
                        0x0F,
                        0xB0,
                        0x54,
                        0xBB,
                        0x16,
                    ],
                ]
            )
            st.session_state.constructed_sbox = aes_sbox
            st.session_state.sbox_name = "AES S-box"
            st.rerun()

        return

    # Get S-box
    sbox = st.session_state.constructed_sbox
    sbox_name = st.session_state.get("sbox_name", "Custom S-box")

    st.success(f"✅ Testing S-box: **{sbox_name}**")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Quick Summary",
            "🔬 Detailed Tests",
            "📈 Visualizations",
            "📋 Compare with Standards",
        ]
    )

    # Run comprehensive test
    with st.spinner("Running cryptographic tests..."):
        results = tester.comprehensive_test(sbox)
        st.session_state.test_results = results

    # Tab 1: Quick Summary
    with tab1:
        st.subheader("Test Results Summary")

        # Display metrics in cards - Row 1
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Nonlinearity (NL)",
                results["NL"]["value"],
                delta=f"Ideal: 112",
                delta_color="off",
            )
            if results["NL"]["value"] >= 112:
                st.success("✅ Excellent")
            elif results["NL"]["value"] >= 100:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col2:
            sac_val = results["SAC"]["value"]
            sac_dev = abs(sac_val - 0.5)
            st.metric(
                "SAC",
                f"{sac_val:.5f}",
                delta=f"Dev: {sac_dev:.5f}",
                delta_color="inverse",
            )
            if sac_dev <= 0.01:
                st.success("✅ Excellent")
            elif sac_dev <= 0.05:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col3:
            st.metric(
                "BIC-NL",
                results["BIC-NL"]["value"],
                delta=f"Ideal: 112",
                delta_color="off",
            )
            if results["BIC-NL"]["value"] >= 112:
                st.success("✅ Excellent")
            elif results["BIC-NL"]["value"] >= 100:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col4:
            bic_sac_val = results["BIC-SAC"]["value"]
            bic_sac_dev = abs(bic_sac_val - 0.5)
            st.metric(
                "BIC-SAC",
                f"{bic_sac_val:.5f}",
                delta=f"Dev: {bic_sac_dev:.5f}",
                delta_color="inverse",
            )
            if bic_sac_dev <= 0.01:
                st.success("✅ Excellent")
            elif bic_sac_dev <= 0.05:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        # Row 2
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "LAP",
                f"{results['LAP']['value']:.6f}",
                delta=f"Ideal: ≤0.0625",
                delta_color="off",
            )
            if results["LAP"]["value"] <= 0.0625:
                st.success("✅ Excellent")
            elif results["LAP"]["value"] <= 0.1:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col6:
            st.metric(
                "DAP",
                f"{results['DAP']['value']:.6f}",
                delta=f"Ideal: ≤0.0156",
                delta_color="off",
            )
            if results["DAP"]["value"] <= 0.015625:
                st.success("✅ Excellent")
            elif results["DAP"]["value"] <= 0.03:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col7:
            st.metric(
                "DU", results["DU"]["value"], delta=f"Ideal: ≤4", delta_color="off"
            )
            if results["DU"]["value"] <= 4:
                st.success("✅ Excellent")
            elif results["DU"]["value"] <= 6:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col8:
            st.metric(
                "AD", results["AD"]["value"], delta=f"Ideal: 7", delta_color="off"
            )
            if results["AD"]["value"] >= 7:
                st.success("✅ Excellent")
            elif results["AD"]["value"] >= 5:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        # Row 3
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            st.metric(
                "TO",
                f"{results['TO']['value']:.5f}",
                delta=f"Ideal: ≈0",
                delta_color="off",
            )
            if results["TO"]["value"] <= 0.1:
                st.success("✅ Excellent")
            elif results["TO"]["value"] <= 0.3:
                st.info("ℹ️ Good")
            else:
                st.warning("⚠️ Needs improvement")

        with col10:
            st.metric(
                "CI",
                results["CI"]["value"],
                delta=f"Ideal: 0",
                delta_color="off",
            )
            if results["CI"]["value"] >= 0:
                st.success("✅ Excellent")
            else:
                st.warning("⚠️ Needs improvement")

        # Overall assessment
        st.write("---")
        st.subheader("Overall Assessment")

        # Calculate strength value (from paper: Equation 20)
        sv = (
            (120 - results["NL"]["value"])
            + abs(0.5 - results["SAC"]["value"])
            + (120 - results["BIC-NL"]["value"])
            + abs(0.5 - results["BIC-SAC"]["value"])
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Strength Value (SV)",
                f"{sv:.6f}",
                help="Lower is better. Ideal is 0. From paper equation (20)",
            )

        with col2:
            # Extended strength score including new metrics
            extended_score = (
                sv
                + (results["DU"]["value"] - 4)
                + (7 - results["AD"]["value"])
                + results["TO"]["value"]
            )
            st.metric(
                "Extended Score",
                f"{extended_score:.6f}",
                help="Includes DU, AD, TO. Lower is better.",
            )

        col1, col2 = st.columns(2)

        with col1:
            # Count excellent criteria
            excellent_count = sum(
                [
                    results["NL"]["value"] >= 112,
                    abs(results["SAC"]["value"] - 0.5) <= 0.01,
                    results["BIC-NL"]["value"] >= 112,
                    abs(results["BIC-SAC"]["value"] - 0.5) <= 0.01,
                    results["LAP"]["value"] <= 0.0625,
                    results["DAP"]["value"] <= 0.015625,
                    results["DU"]["value"] <= 4,
                    results["AD"]["value"] >= 7,
                    results["TO"]["value"] <= 0.1,
                    results["CI"]["value"] >= 0,
                ]
            )

            if excellent_count >= 9:
                st.success(
                    f"🎉 **EXCELLENT S-box!** {excellent_count}/10 criteria met!"
                )
            elif excellent_count >= 7:
                st.info(f"✅ **GOOD S-box** - {excellent_count}/10 criteria excellent")
            else:
                st.warning(
                    f"⚠️ **Moderate S-box** - {excellent_count}/10 criteria excellent"
                )

        with col2:
            # Download results
            results_text = f"""S-box Test Results
==================

Core Metrics (from paper):
--------------------------
Nonlinearity (NL): {results['NL']['value']}
SAC: {results['SAC']['value']:.6f}
BIC-NL: {results['BIC-NL']['value']}
BIC-SAC: {results['BIC-SAC']['value']:.6f}
LAP: {results['LAP']['value']:.6f}
DAP: {results['DAP']['value']:.6f}

Additional Metrics:
-------------------
Differential Uniformity (DU): {results['DU']['value']}
Algebraic Degree (AD): {results['AD']['value']}
Transparency Order (TO): {results['TO']['value']:.6f}
Correlation Immunity (CI): {results['CI']['value']}

Strength Values:
----------------
SV (Paper): {sv:.6f}
Extended Score: {extended_score:.6f}
Excellent Criteria: {excellent_count}/10
"""
            st.download_button(
                "📥 Download Results",
                data=results_text,
                file_name="sbox_test_results.txt",
                mime="text/plain",
                width="stretch",
            )

    # Tab 2: Detailed Tests
    with tab2:
        st.subheader("Detailed Test Results")

        test_choice = st.selectbox(
            "Select test to view details:",
            [
                "Nonlinearity (NL)",
                "SAC",
                "BIC-NL",
                "BIC-SAC",
                "LAP",
                "DAP",
                "Differential Uniformity (DU)",
                "Algebraic Degree (AD)",
                "Transparency Order (TO)",
                "Correlation Immunity (CI)",
            ],
        )

        if test_choice == "Nonlinearity (NL)":
            st.write("### Nonlinearity Test")
            st.write(
                """
            **Nonlinearity** measures the minimum distance between the Boolean functions 
            representing the S-box and all affine functions. Higher values indicate better 
            resistance to linear cryptanalysis.
            
            **Ideal value:** 112 for 8-bit S-boxes
            """
            )

            nl_details = results["NL"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall NL", nl_details["overall"])
            col2.metric("Min NL", nl_details["min"])
            col3.metric("Max NL", nl_details["max"])

            # Per-bit NL values
            st.write("**Nonlinearity per output bit:**")
            nl_per_bit = pd.DataFrame(
                {
                    "Output Bit": [f"f_{i}" for i in range(8)],
                    "NL Value": [nl_details["per_bit"][f"f_{i}"] for i in range(8)],
                    "Status": [
                        "✅" if nl_details["per_bit"][f"f_{i}"] >= 112 else "⚠️"
                        for i in range(8)
                    ],
                }
            )
            st.dataframe(nl_per_bit, width="stretch", hide_index=True)

        elif test_choice == "SAC":
            st.write("### Strict Avalanche Criterion Test")
            st.write(
                """
            **SAC** measures how a single bit change in input affects the output bits.
            Each output bit should change with probability 0.5 when any input bit is flipped.
            
            **Ideal value:** 0.5
            """
            )

            sac_details = results["SAC"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall SAC", f"{sac_details['overall']:.6f}")
            col2.metric("Deviation", f"{sac_details['deviation']:.6f}")
            col3.metric("Std Dev", f"{sac_details['std']:.6f}")

            # SAC Matrix
            st.write("**SAC Matrix (Input Bit × Output Bit):**")
            sac_matrix_df = pd.DataFrame(
                sac_details["matrix"],
                columns=[f"Out{i}" for i in range(8)],
                index=[f"In{i}" for i in range(8)],
            )
            st.dataframe(sac_matrix_df.style.format("{:.4f}"), width="stretch")

        elif test_choice == "BIC-NL":
            st.write("### Bit Independence Criterion - Nonlinearity Test")
            st.write(
                """
            **BIC-NL** measures the nonlinearity of XOR combinations of output bit pairs.
            It ensures that different output bits are independent.
            
            **Ideal value:** 112 for 8-bit S-boxes
            """
            )

            bic_nl_details = results["BIC-NL"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall BIC-NL", bic_nl_details["overall"])
            col2.metric("Min BIC-NL", bic_nl_details["min"])
            col3.metric("Max BIC-NL", bic_nl_details["max"])

            # BIC-NL Matrix
            st.write("**BIC-NL Matrix (Output Bit Pairs):**")
            bic_nl_matrix_df = pd.DataFrame(
                bic_nl_details["matrix"],
                columns=[f"f{i}" for i in range(8)],
                index=[f"f{i}" for i in range(8)],
            )
            st.dataframe(bic_nl_matrix_df.style.format("{:.0f}"), width="stretch")

        elif test_choice == "BIC-SAC":
            st.write("### Bit Independence Criterion - SAC Test")
            st.write(
                """
            **BIC-SAC** measures the independence of the avalanche effect between 
            different output bit pairs.
            
            **Ideal value:** 0.5
            """
            )

            bic_sac_details = results["BIC-SAC"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall BIC-SAC", f"{bic_sac_details['overall']:.6f}")
            col2.metric("Deviation", f"{bic_sac_details['deviation']:.6f}")
            col3.metric("Std Dev", f"{bic_sac_details['std']:.6f}")

            # BIC-SAC Matrix
            st.write("**BIC-SAC Matrix (Output Bit Pairs):**")
            bic_sac_matrix_df = pd.DataFrame(
                bic_sac_details["matrix"],
                columns=[f"f{i}" for i in range(8)],
                index=[f"f{i}" for i in range(8)],
            )
            st.dataframe(bic_sac_matrix_df.style.format("{:.5f}"), width="stretch")

        elif test_choice == "LAP":
            st.write("### Linear Approximation Probability Test")
            st.write(
                """
            **LAP** measures the maximum probability of a linear approximation.
            Lower values indicate better resistance to linear cryptanalysis.
            
            **Ideal value:** 0.0625 (1/16) for 8-bit S-boxes
            """
            )

            lap_details = results["LAP"]["details"]

            col1, col2 = st.columns(2)
            col1.metric("LAP", f"{lap_details['lap']:.6f}")
            col2.metric("Max Bias", f"{lap_details['max_bias']:.6f}")

            st.write("**Best Linear Approximation:**")
            st.code(
                f"""
Input Mask:  {lap_details['best_input_mask']}
Output Mask: {lap_details['best_output_mask']}
Bias:        {lap_details['max_bias']:.6f}
            """
            )

        elif test_choice == "DAP":
            st.write("### Differential Approximation Probability Test")
            st.write(
                """
            **DAP** measures the maximum differential probability.
            Lower values indicate better resistance to differential cryptanalysis.
            
            **Ideal value:** 0.015625 (1/64) for 8-bit S-boxes
            """
            )

            dap_details = results["DAP"]["details"]

            col1, col2 = st.columns(2)
            col1.metric("DAP", f"{dap_details['dap']:.6f}")
            col2.metric("Max Count", dap_details["max_count"])

            st.write("**Best Differential:**")
            st.code(
                f"""
Input Difference:  {dap_details['best_input_diff']}
Output Difference: {dap_details['best_output_diff']}
Occurrences:       {dap_details['max_count']} / 256
            """
            )

        elif test_choice == "Differential Uniformity (DU)":
            st.write("### Differential Uniformity Test")
            st.write(
                """
            **Differential Uniformity** is the maximum entry in the Difference Distribution Table.
            It measures the worst-case differential probability for the S-box.
            Lower values indicate better resistance to differential cryptanalysis.
            
            **Ideal value:** 4 for AES-like S-boxes (DU=4 is considered optimal)
            """
            )

            du_details = results["DU"]["details"]

            col1, col2 = st.columns(2)
            col1.metric("DU Value", du_details["du"])
            col2.metric("Ideal", du_details["ideal"])

            st.write("**Best Differential:**")
            st.code(
                f"""
Input Difference:  {du_details['best_input_diff']}
Output Difference: {du_details['best_output_diff']}
Max Count:         {du_details['du']}
            """
            )

            if du_details["du"] <= 4:
                st.success("✅ Excellent differential uniformity!")
            elif du_details["du"] <= 6:
                st.info("ℹ️ Good differential uniformity")
            else:
                st.warning("⚠️ High differential uniformity - may be vulnerable")

        elif test_choice == "Algebraic Degree (AD)":
            st.write("### Algebraic Degree Test")
            st.write(
                """
            **Algebraic Degree** is the degree of the polynomial representation of the S-box.
            Higher degrees provide better resistance to algebraic attacks.
            
            **Ideal value:** 7 (maximum for 8-bit S-boxes is n-1 = 7)
            """
            )

            ad_details = results["AD"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Min Degree", ad_details["min_degree"])
            col2.metric("Max Degree", ad_details["max_degree"])
            col3.metric("Avg Degree", f"{ad_details['avg_degree']:.2f}")

            # Per-bit degrees
            st.write("**Algebraic Degree per output bit:**")
            ad_per_bit = pd.DataFrame(
                {
                    "Output Bit": [f"f_{i}" for i in range(8)],
                    "Degree": [ad_details["per_bit"][f"f_{i}"] for i in range(8)],
                    "Status": [
                        "✅" if ad_details["per_bit"][f"f_{i}"] >= 7 else "⚠️"
                        for i in range(8)
                    ],
                }
            )
            st.dataframe(ad_per_bit, width="stretch", hide_index=True)

        elif test_choice == "Transparency Order (TO)":
            st.write("### Transparency Order Test")
            st.write(
                """
            **Transparency Order** measures the correlation between input and output bits.
            Lower values indicate better confusion property - the S-box should obscure
            the relationship between input and output.
            
            **Ideal value:** Close to 0 (perfect confusion)
            """
            )

            to_details = results["TO"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Max TO", f"{to_details['max_to']:.5f}")
            col2.metric("Avg TO", f"{to_details['avg_to']:.5f}")
            col3.metric("Min TO", f"{to_details['min_to']:.5f}")

            # TO Matrix
            st.write("**Transparency Matrix (Input Bit × Output Bit):**")
            to_matrix_df = pd.DataFrame(
                to_details["matrix"],
                columns=[f"Out{i}" for i in range(8)],
                index=[f"In{i}" for i in range(8)],
            )
            st.dataframe(to_matrix_df.style.format("{:.4f}"), width="stretch")

        elif test_choice == "Correlation Immunity (CI)":
            st.write("### Correlation Immunity Test")
            st.write(
                """
            **Correlation Immunity** measures resistance to correlation attacks.
            A function has CI of order m if its output is statistically independent
            of any m input variables. Higher values are better.
            
            **Higher is better** - indicates better resistance to correlation attacks.
            """
            )

            ci_details = results["CI"]["details"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Min CI", ci_details["min_ci"])
            col2.metric("Max CI", ci_details["max_ci"])
            col3.metric("Avg CI", f"{ci_details['avg_ci']:.2f}")

            # Per-bit CI
            st.write("**Correlation Immunity per output bit:**")
            ci_per_bit = pd.DataFrame(
                {
                    "Output Bit": [f"f_{i}" for i in range(8)],
                    "CI Order": [ci_details["per_bit"][f"f_{i}"] for i in range(8)],
                    "Status": [
                        ("✅" if ci_details["per_bit"][f"f_{i}"] >= 0 else "ℹ️")
                        for i in range(8)
                    ],
                }
            )
            st.dataframe(ci_per_bit, width="stretch", hide_index=True)

    # Tab 3: Visualizations
    with tab3:
        st.subheader("Test Results Visualizations")

        viz_choice = st.selectbox(
            "Select visualization:",
            [
                "Overview Radar Chart",
                "SAC Heatmap",
                "BIC-SAC Heatmap",
                "Transparency Order Heatmap",
                "Comparison Bar Chart",
            ],
        )

        if viz_choice == "Overview Radar Chart":
            st.write("**Cryptographic Strength Radar Chart**")

            # Normalize values for radar chart
            metrics = {
                "NL": results["NL"]["value"] / 112.0,
                "SAC": 1 - abs(results["SAC"]["value"] - 0.5) * 2,
                "BIC-NL": results["BIC-NL"]["value"] / 112.0,
                "BIC-SAC": 1 - abs(results["BIC-SAC"]["value"] - 0.5) * 2,
                "LAP": 1 - (results["LAP"]["value"] / 0.125),
                "DAP": 1 - (results["DAP"]["value"] / 0.03125),
                "DU": 1 - (results["DU"]["value"] / 8.0),  # Normalize: lower is better
                "AD": results["AD"]["value"] / 7.0,  # Normalize: higher is better
                "TO": 1 - min(results["TO"]["value"], 1.0),  # Lower is better
                "CI": min(results["CI"]["value"] / 5.0, 1.0),  # Higher is better
            }

            # Ensure values are between 0 and 1
            for k in metrics:
                metrics[k] = max(0, min(1, metrics[k]))

            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )

            categories = list(metrics.keys())
            values = list(metrics.values())
            values += values[:1]  # Complete the circle

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            ax.plot(angles, values, "o-", linewidth=2, label=sbox_name, color="#2E86AB")
            ax.fill(angles, values, alpha=0.25, color="#2E86AB")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["25%", "50%", "75%", "100%"])
            ax.grid(True)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.set_title(
                "S-box Cryptographic Strength Profile",
                pad=20,
                fontsize=14,
                fontweight="bold",
            )

            st.pyplot(fig)
            plt.close()

        elif viz_choice == "Transparency Order Heatmap":
            st.write("**Transparency Order Matrix Heatmap**")

            fig, ax = plt.subplots(figsize=(10, 8))
            to_matrix = results["TO"]["details"]["matrix"]

            sns.heatmap(
                to_matrix,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn_r",
                center=0.5,
                vmin=0,
                vmax=1,
                xticklabels=[f"Out{i}" for i in range(8)],
                yticklabels=[f"In{i}" for i in range(8)],
                ax=ax,
                cbar_kws={"label": "Correlation"},
            )
            ax.set_title("Transparency Order Matrix: Input-Output Bit Correlation")
            ax.set_xlabel("Output Bit")
            ax.set_ylabel("Input Bit")

            st.pyplot(fig)
            plt.close()

            st.info(
                "💡 **Interpretation:** Green (low values) = good confusion. Red (high values) = high correlation."
            )

            st.info(
                "💡 **Interpretation:** Larger area = stronger S-box. All metrics normalized to 0-1 scale."
            )

        elif viz_choice == "SAC Heatmap":
            st.write("**SAC Matrix Heatmap**")

            fig, ax = plt.subplots(figsize=(10, 8))
            sac_matrix = results["SAC"]["details"]["matrix"]

            sns.heatmap(
                sac_matrix,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                center=0.5,
                vmin=0,
                vmax=1,
                xticklabels=[f"Out{i}" for i in range(8)],
                yticklabels=[f"In{i}" for i in range(8)],
                ax=ax,
                cbar_kws={"label": "Probability"},
            )
            ax.set_title("SAC Matrix: Input Bit vs Output Bit Change Probability")
            ax.set_xlabel("Output Bit")
            ax.set_ylabel("Input Bit")

            st.pyplot(fig)
            plt.close()

        elif viz_choice == "BIC-SAC Heatmap":
            st.write("**BIC-SAC Matrix Heatmap**")

            fig, ax = plt.subplots(figsize=(10, 8))
            bic_sac_matrix = results["BIC-SAC"]["details"]["matrix"]

            sns.heatmap(
                bic_sac_matrix,
                annot=True,
                fmt=".4f",
                cmap="RdYlGn",
                center=0.5,
                vmin=0.4,
                vmax=0.6,
                xticklabels=[f"f{i}" for i in range(8)],
                yticklabels=[f"f{i}" for i in range(8)],
                ax=ax,
                cbar_kws={"label": "Independence"},
            )
            ax.set_title("BIC-SAC Matrix: Output Bit Pair Independence")
            ax.set_xlabel("Output Bit")
            ax.set_ylabel("Output Bit")

            st.pyplot(fig)
            plt.close()

        elif viz_choice == "Comparison Bar Chart":
            st.write("**Test Results vs Ideal Values**")

            # Prepare data for 10 tests
            tests = [
                "NL",
                "SAC",
                "BIC-NL",
                "BIC-SAC",
                "LAP",
                "DAP",
                "DU",
                "AD",
                "TO",
                "CI",
            ]
            actual = [
                results["NL"]["value"],
                results["SAC"]["value"],
                results["BIC-NL"]["value"],
                results["BIC-SAC"]["value"],
                results["LAP"]["value"],
                results["DAP"]["value"],
                results["DU"]["value"],
                results["AD"]["value"],
                results["TO"]["value"],
                results["CI"]["value"],
            ]
            ideal = [112, 0.5, 112, 0.5, 0.0625, 0.015625, 4, 7, 0.0, 5]

            # Create subplots
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.flatten()

            for i, (test, act, idl) in enumerate(zip(tests, actual, ideal)):
                if i < len(axes):
                    ax = axes[i]

                    if test in ["NL", "BIC-NL", "DU", "AD", "CI"]:
                        # Bar chart for integer metrics
                        ax.bar(
                            ["Actual", "Ideal"],
                            [act, idl],
                            color=["#2E86AB", "#A23B72"],
                        )
                        if test in ["NL", "BIC-NL", "AD", "CI"]:
                            ax.set_ylim(0, max(act, idl) + 20)
                        else:  # DU
                            ax.set_ylim(0, max(act, idl) + 2)
                        ax.set_ylabel("Value")
                    else:
                        # Bar chart for probability tests
                        ax.bar(
                            ["Actual", "Ideal"],
                            [act, idl],
                            color=["#2E86AB", "#A23B72"],
                        )
                        ax.set_ylabel("Value")

                    ax.set_title(f"{test}", fontweight="bold")
                    ax.grid(axis="y", alpha=0.3)

                    # Add value labels
                    for j, (label, value) in enumerate(
                        zip(["Actual", "Ideal"], [act, idl])
                    ):
                        if test in ["NL", "BIC-NL", "DU", "AD", "CI"]:
                            ax.text(
                                j,
                                value,
                                f"{value:.0f}",
                                ha="center",
                                va="bottom",
                                fontweight="bold",
                            )
                        else:
                            ax.text(
                                j,
                                value,
                                f"{value:.4f}",
                                ha="center",
                                va="bottom",
                                fontweight="bold",
                                fontsize=9,
                            )

            # Hide unused subplots
            for i in range(len(tests), len(axes)):
                axes[i].axis("off")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 4: Compare with Standards
    with tab4:
        st.subheader("Compare with Standard S-boxes")

        st.write(
            """
        Compare your S-box with well-known S-boxes from literature:
        - **AES S-box**: Standard AES substitution box
        - **Paper S-box 44**: The proposed S-box from the paper
        """
        )

        # Comparison data - extended with new metrics
        comparison_data = {
            "Metric": [
                "NL",
                "SAC",
                "BIC-NL",
                "BIC-SAC",
                "LAP",
                "DAP",
                "DU",
                "AD",
                "TO",
                "CI",
                "SV",
            ],
            "Your S-box": [
                results["NL"]["value"],
                results["SAC"]["value"],
                results["BIC-NL"]["value"],
                results["BIC-SAC"]["value"],
                results["LAP"]["value"],
                results["DAP"]["value"],
                results["DU"]["value"],
                results["AD"]["value"],
                results["TO"]["value"],
                results["CI"]["value"],
                (120 - results["NL"]["value"])
                + abs(0.5 - results["SAC"]["value"])
                + (120 - results["BIC-NL"]["value"])
                + abs(0.5 - results["BIC-SAC"]["value"]),
            ],
            "AES S-box": [
                112,
                0.50488,
                112,
                0.50460,
                0.0625,
                0.01563,
                4,
                7,
                0.25,
                0,
                16.00948,
            ],
            "Paper S-box 44": [
                112,
                0.50073,
                112,
                0.50237,
                0.0625,
                0.01563,
                4,
                7,
                0.25,
                0,
                16.0031,
            ],
            "Ideal": [
                112,
                0.5,
                112,
                0.5,
                0.0625,
                0.015625,
                4,
                7,
                0.0,
                0,
                0.0,
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)

        def fmt(x):
            try:
                v = float(x)
            except (ValueError, TypeError):
                return x

            if v < 1:
                return f"{v:.6f}"
            elif v > 10:
                return f"{v:.0f}"
            else:
                return f"{v:.5f}"

        # Format the dataframe
        styled_df = comparison_df.style.format(
            {
                "Your S-box": fmt,
                "AES S-box": fmt,
                "Paper S-box 44": fmt,
                "Ideal": fmt,
            }
        )

        st.dataframe(styled_df, width="stretch", hide_index=True)

        st.write("---")
        st.write("**Performance Analysis:**")

        # Check if better than AES
        better_count = 0
        equal_count = 0

        if results["NL"]["value"] >= 112:
            equal_count += 1
        if abs(results["SAC"]["value"] - 0.5) < abs(0.50488 - 0.5):
            better_count += 1
        if results["BIC-NL"]["value"] >= 112:
            equal_count += 1
        if abs(results["BIC-SAC"]["value"] - 0.5) < abs(0.50460 - 0.5):
            better_count += 1
        if results["LAP"]["value"] <= 0.0625:
            equal_count += 1
        if results["DAP"]["value"] <= 0.01563:
            equal_count += 1

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**{better_count}** metrics better than AES")
            st.info(f"**{equal_count}** metrics equal to ideal")

        with col2:
            sv_your = (
                (120 - results["NL"]["value"])
                + abs(0.5 - results["SAC"]["value"])
                + (120 - results["BIC-NL"]["value"])
                + abs(0.5 - results["BIC-SAC"]["value"])
            )
            sv_aes = 16.00948
            sv_paper = 16.0031

            if sv_your < sv_paper:
                st.success(f"🎉 **Your S-box is BETTER than Paper S-box 44!**")
                st.metric("SV Improvement", f"{sv_paper - sv_your:.6f}")
            elif sv_your < sv_aes:
                st.success(f"✅ **Your S-box is better than AES!**")
                st.metric("SV Improvement", f"{sv_aes - sv_your:.6f}")
            else:
                st.info(f"ℹ️ **Room for improvement**")
                st.metric("SV Difference from Paper", f"{sv_your - sv_paper:.6f}")


# Main execution for standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="S-box Tester", page_icon="🧪", layout="wide")

    render_sbox_tester()
