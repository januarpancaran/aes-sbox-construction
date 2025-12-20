"""
AES Implementation with Custom S-box
Complete encrypt/decrypt interface for text and images using custom or standard S-boxes
"""

import os
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt


# AES Implementation (from GitHub with S-box modification support)
def bytes2matrix(text):
    """Converts a 16-byte array into a 4x4 matrix."""
    return [list(text[i : i + 4]) for i in range(0, len(text), 4)]


def matrix2bytes(matrix):
    """Converts a 4x4 matrix into a 16-byte array."""
    return bytes(sum(matrix, []))


def xor_bytes(a, b):
    """Returns a new byte array with the elements xor'ed."""
    return bytes(i ^ j for i, j in zip(a, b))


def pad(plaintext):
    """Pads the given plaintext with PKCS#7 padding."""
    padding_len = 16 - (len(plaintext) % 16)
    padding = bytes([padding_len] * padding_len)
    return plaintext + padding


def pad_key(key):
    """Pads or truncates key to exactly 16 bytes."""
    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    if len(key_bytes) < 16:
        # Pad with zeros
        return key_bytes + b"\x00" * (16 - len(key_bytes))
    elif len(key_bytes) > 16:
        # Truncate to 16 bytes
        return key_bytes[:16]
    return key_bytes


def unpad(plaintext):
    """Removes a PKCS#7 padding."""
    padding_len = plaintext[-1]
    return plaintext[:-padding_len]


xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


class AES:
    """AES-128 encryption with customizable S-box."""

    # Standard AES S-box
    STANDARD_SBOX = (
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
    )

    r_con = (
        0x00,
        0x01,
        0x02,
        0x04,
        0x08,
        0x10,
        0x20,
        0x40,
        0x80,
        0x1B,
        0x36,
        0x6C,
        0xD8,
        0xAB,
        0x4D,
        0x9A,
        0x2F,
        0x5E,
        0xBC,
        0x63,
        0xC6,
        0x97,
        0x35,
        0x6A,
        0xD4,
        0xB3,
        0x7D,
        0xFA,
        0xEF,
        0xC5,
        0x91,
        0x39,
    )

    def __init__(self, master_key, custom_sbox=None):
        """Initialize AES with 16-byte key and optional custom S-box."""
        assert len(master_key) == 16
        self.n_rounds = 10

        # Set S-box
        if custom_sbox is not None:
            self.s_box = tuple(custom_sbox.flatten().tolist())
        else:
            self.s_box = self.STANDARD_SBOX

        # Generate inverse S-box
        self.inv_s_box = [0] * 256
        for i in range(256):
            self.inv_s_box[self.s_box[i]] = i
        self.inv_s_box = tuple(self.inv_s_box)

        # Expand key
        self._key_matrices = self._expand_key(master_key)

    def _expand_key(self, master_key):
        """Expands the master key into round keys."""
        key_columns = bytes2matrix(master_key)
        iteration_size = 4

        i = 1
        while len(key_columns) < (self.n_rounds + 1) * 4:
            word = list(key_columns[-1])

            if len(key_columns) % iteration_size == 0:
                word.append(word.pop(0))
                word = [self.s_box[b] for b in word]
                word[0] ^= self.r_con[i]
                i += 1

            word = xor_bytes(word, key_columns[-iteration_size])
            key_columns.append(word)

        return [key_columns[4 * i : 4 * (i + 1)] for i in range(len(key_columns) // 4)]

    def _sub_bytes(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = self.s_box[s[i][j]]

    def _inv_sub_bytes(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = self.inv_s_box[s[i][j]]

    def _shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]

    def _inv_shift_rows(self, s):
        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def _mix_single_column(self, a):
        t = a[0] ^ a[1] ^ a[2] ^ a[3]
        u = a[0]
        a[0] ^= t ^ xtime(a[0] ^ a[1])
        a[1] ^= t ^ xtime(a[1] ^ a[2])
        a[2] ^= t ^ xtime(a[2] ^ a[3])
        a[3] ^= t ^ xtime(a[3] ^ u)

    def _mix_columns(self, s):
        for i in range(4):
            self._mix_single_column(s[i])

    def _inv_mix_columns(self, s):
        for i in range(4):
            u = xtime(xtime(s[i][0] ^ s[i][2]))
            v = xtime(xtime(s[i][1] ^ s[i][3]))
            s[i][0] ^= u
            s[i][1] ^= v
            s[i][2] ^= u
            s[i][3] ^= v
        self._mix_columns(s)

    def _add_round_key(self, s, k):
        for i in range(4):
            for j in range(4):
                s[i][j] ^= k[i][j]

    def encrypt_block(self, plaintext):
        """Encrypts a single block of 16 bytes."""
        assert len(plaintext) == 16

        plain_state = bytes2matrix(plaintext)
        self._add_round_key(plain_state, self._key_matrices[0])

        for i in range(1, self.n_rounds):
            self._sub_bytes(plain_state)
            self._shift_rows(plain_state)
            self._mix_columns(plain_state)
            self._add_round_key(plain_state, self._key_matrices[i])

        self._sub_bytes(plain_state)
        self._shift_rows(plain_state)
        self._add_round_key(plain_state, self._key_matrices[-1])

        return matrix2bytes(plain_state)

    def decrypt_block(self, ciphertext):
        """Decrypts a single block of 16 bytes."""
        assert len(ciphertext) == 16

        cipher_state = bytes2matrix(ciphertext)
        self._add_round_key(cipher_state, self._key_matrices[-1])
        self._inv_shift_rows(cipher_state)
        self._inv_sub_bytes(cipher_state)

        for i in range(self.n_rounds - 1, 0, -1):
            self._add_round_key(cipher_state, self._key_matrices[i])
            self._inv_mix_columns(cipher_state)
            self._inv_shift_rows(cipher_state)
            self._inv_sub_bytes(cipher_state)

        self._add_round_key(cipher_state, self._key_matrices[0])

        return matrix2bytes(cipher_state)

    def encrypt_cbc(self, plaintext, iv):
        """Encrypts using CBC mode with PKCS#7 padding."""
        assert len(iv) == 16
        plaintext = pad(plaintext)

        blocks = []
        previous = iv
        for i in range(0, len(plaintext), 16):
            block = plaintext[i : i + 16]
            block = xor_bytes(block, previous)
            encrypted = self.encrypt_block(block)
            blocks.append(encrypted)
            previous = encrypted

        return b"".join(blocks)

    def decrypt_cbc(self, ciphertext, iv):
        """Decrypts using CBC mode with PKCS#7 padding."""
        assert len(iv) == 16

        blocks = []
        previous = iv
        for i in range(0, len(ciphertext), 16):
            block = ciphertext[i : i + 16]
            decrypted = self.decrypt_block(block)
            decrypted = xor_bytes(decrypted, previous)
            blocks.append(decrypted)
            previous = block

        return unpad(b"".join(blocks))


def render_aes_implementation():
    """Streamlit component for AES encryption/decryption."""
    st.header("🔐 AES Encryption/Decryption")

    # Mode selection tabs
    tab1, tab2, tab3 = st.tabs(
        ["📝 Text Encryption", "🖼️ Image Encryption", "📊 Image Analysis"]
    )

    with tab1:
        render_text_encryption()

    with tab2:
        render_image_encryption()

    with tab3:
        render_image_analysis()


def render_text_encryption():
    """Original text encryption interface (extracted from main function)."""
    st.info(
        """
    **Simple AES-128 Text Encryption**
    
    Encrypt and decrypt text messages using standard AES or your custom S-boxes.
    Uses CBC mode with random IV for security.
    """
    )

    # Select S-box
    st.subheader("1️⃣ Select S-box")

    sbox_options = ["Standard AES S-box"]

    # Add saved S-boxes
    if hasattr(st.session_state, "saved_sboxes") and st.session_state.saved_sboxes:
        sbox_options.extend(list(st.session_state.saved_sboxes.keys()))

    # Add current S-box
    if hasattr(st.session_state, "constructed_sbox"):
        current_name = st.session_state.get("sbox_name", "Current S-box")
        if current_name not in sbox_options:
            sbox_options.insert(1, current_name)

    selected_sbox = st.selectbox("Choose S-box:", options=sbox_options)

    # Get the S-box
    if selected_sbox == "Standard AES S-box":
        sbox = None
        st.success("✅ Using standard AES S-box")
    elif selected_sbox == st.session_state.get("sbox_name", "Current S-box"):
        sbox = st.session_state.constructed_sbox
        st.success(f"✅ Using: {selected_sbox}")
    else:
        sbox = st.session_state.saved_sboxes[selected_sbox]["sbox"]
        st.success(f"✅ Using: {selected_sbox}")

    st.write("---")

    # Two columns for encrypt and decrypt
    col1, col2 = st.columns(2)

    # ENCRYPT COLUMN
    with col1:
        st.subheader("🔒 Encrypt")

        # Key input
        encrypt_key = st.text_input(
            "Encryption Key (any length):",
            value="MySecretKey12345",
            key="enc_key",
        )

        # Show padded key length
        key_info = f"Key length: {len(encrypt_key)} chars"
        if len(encrypt_key) < 16:
            key_info += f" → will be padded to 16 bytes"
        elif len(encrypt_key) > 16:
            key_info += f" → will be truncated to 16 bytes"
        else:
            key_info += " ✓ (exactly 16 bytes)"
        st.caption(key_info)

        # Message input
        plaintext = st.text_area(
            "Message to encrypt:",
            value="Hello, World! This is a secret message.",
            height=150,
            key="plaintext",
        )

        # Encrypt button
        if st.button("🔒 Encrypt", width="stretch", type="primary"):
            if not encrypt_key:
                st.error("❌ Please enter an encryption key")
            elif not plaintext:
                st.error("❌ Please enter a message")
            else:
                try:
                    # Generate random IV
                    iv = os.urandom(16)

                    # Encrypt
                    key_bytes = pad_key(encrypt_key)
                    message_bytes = plaintext.encode("utf-8")

                    aes = AES(key_bytes, sbox)
                    ciphertext = aes.encrypt_cbc(message_bytes, iv)

                    # Combine IV + ciphertext
                    result = iv + ciphertext

                    st.success("✅ Encrypted successfully!")

                    # Display result
                    st.write("**Encrypted (Hex):**")
                    st.code(result.hex(), language="text")

                    # Stats
                    st.metric("Size", f"{len(result)} bytes")

                    # Store for decrypt
                    st.session_state.last_encrypted = result.hex()
                    st.session_state.last_key = encrypt_key

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # DECRYPT COLUMN
    with col2:
        st.subheader("🔓 Decrypt")

        # Key input
        decrypt_key = st.text_input(
            "Decryption Key (any length):",
            value=st.session_state.get("last_key", "MySecretKey12345"),
            key="dec_key",
        )

        # Show padded key length
        key_info = f"Key length: {len(decrypt_key)} chars"
        if len(decrypt_key) < 16:
            key_info += f" → will be padded to 16 bytes"
        elif len(decrypt_key) > 16:
            key_info += f" → will be truncated to 16 bytes"
        else:
            key_info += " ✓ (exactly 16 bytes)"
        st.caption(key_info)

        # Ciphertext input
        ciphertext_hex = st.text_area(
            "Encrypted message (hex):",
            value=st.session_state.get("last_encrypted", ""),
            height=150,
            placeholder="Paste encrypted hex here...",
            key="ciphertext",
        )

        # Decrypt button
        if st.button("🔓 Decrypt", width="stretch", type="primary"):
            if not decrypt_key:
                st.error("❌ Please enter a decryption key")
            elif not ciphertext_hex:
                st.error("❌ Please enter ciphertext")
            else:
                try:
                    # Parse hex
                    ciphertext = bytes.fromhex(
                        ciphertext_hex.replace(" ", "").replace("\n", "")
                    )

                    if len(ciphertext) < 32:
                        st.error("❌ Ciphertext too short (must include IV)")
                    else:
                        # Extract IV and ciphertext
                        iv = ciphertext[:16]
                        encrypted_data = ciphertext[16:]

                        # Decrypt
                        key_bytes = pad_key(decrypt_key)

                        aes = AES(key_bytes, sbox)
                        decrypted = aes.decrypt_cbc(encrypted_data, iv)

                        st.success("✅ Decrypted successfully!")

                        # Display result
                        st.write("**Decrypted Message:**")
                        try:
                            message = decrypted.decode("utf-8")
                            st.text_area(
                                "Decrypted output",
                                value=message,
                                height=150,
                                key="decrypted_output",
                            )
                        except:
                            st.warning("Cannot decode as text. Showing hex:")
                            st.code(decrypted.hex(), language="text")

                        # Stats
                        st.metric("Size", f"{len(decrypted)} bytes")

                except ValueError:
                    st.error("❌ Invalid hex string")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Help section
    st.write("---")

    with st.expander("ℹ️ How to use Text Encryption"):
        st.markdown(
            """
        ### 📝 Instructions
        
        **To Encrypt:**
        1. Select an S-box (standard or custom)
        2. Enter an encryption key (any length - will be padded/truncated to 16 bytes)
        3. Type your message
        4. Click "Encrypt"
        5. Copy the hex output for decryption
        
        **To Decrypt:**
        1. Use the SAME S-box and key that was used for encryption
        2. Paste the encrypted hex string
        3. Click "Decrypt"
        
        **Key Handling:**
        - Keys shorter than 16 bytes are padded with zeros
        - Keys longer than 16 bytes are truncated to 16 bytes
        - For best security, use exactly 16 characters
        - Example keys: "secret", "MySecretKey12345", "a very long password"
        
        **Security Notes:**
        - The same key and S-box must be used for encryption and decryption
        - Uses CBC mode with random IV (included in output)
        - Standard AES S-box is the proven secure choice
        - Custom S-boxes are experimental - use for testing only
        
        **Example:**
        ```
        Key: MySecretKey12345
        Message: Hello World
        ```
        """
        )


def render_image_encryption():
    """Image encryption/decryption interface."""
    st.info(
        """
    **AES Image Encryption**
    
    Encrypt and decrypt images using standard AES or your custom S-boxes.
    The entire image data is encrypted while preserving dimensions.
    """
    )

    # Select S-box
    st.subheader("1️⃣ Select S-box")

    sbox_options = ["Standard AES S-box"]

    # Add saved S-boxes
    if hasattr(st.session_state, "saved_sboxes") and st.session_state.saved_sboxes:
        sbox_options.extend(list(st.session_state.saved_sboxes.keys()))

    # Add current S-box
    if hasattr(st.session_state, "constructed_sbox"):
        current_name = st.session_state.get("sbox_name", "Current S-box")
        if current_name not in sbox_options:
            sbox_options.insert(1, current_name)

    selected_sbox = st.selectbox("Choose S-box:", options=sbox_options, key="img_sbox")

    # Get the S-box
    if selected_sbox == "Standard AES S-box":
        sbox = None
        st.success("✅ Using standard AES S-box")
    elif selected_sbox == st.session_state.get("sbox_name", "Current S-box"):
        sbox = st.session_state.constructed_sbox
        st.success(f"✅ Using: {selected_sbox}")
    else:
        sbox = st.session_state.saved_sboxes[selected_sbox]["sbox"]
        st.success(f"✅ Using: {selected_sbox}")

    st.write("---")

    # Two columns for encrypt and decrypt
    col1, col2 = st.columns(2)

    # ENCRYPT COLUMN
    with col1:
        st.subheader("🔒 Encrypt Image")

        # Key input
        encrypt_key = st.text_input(
            "Encryption Key (any length):",
            value="MySecretKey12345",
            key="img_enc_key",
        )

        # Show padded key length
        key_info = f"Key length: {len(encrypt_key)} chars"
        if len(encrypt_key) < 16:
            key_info += f" → will be padded to 16 bytes"
        elif len(encrypt_key) > 16:
            key_info += f" → will be truncated to 16 bytes"
        else:
            key_info += " ✓ (exactly 16 bytes)"
        st.caption(key_info)

        # Image upload
        uploaded_file = st.file_uploader(
            "Upload image to encrypt:",
            type=["png", "jpg", "jpeg", "bmp"],
            key="img_upload",
        )

        if uploaded_file:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width="stretch")

            # Image info
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")

        # Encrypt button
        if st.button("🔒 Encrypt Image", type="primary", width="stretch"):
            if not encrypt_key:
                st.error("❌ Please enter an encryption key")
            elif not uploaded_file:
                st.error("❌ Please upload an image")
            else:
                try:
                    with st.spinner("Encrypting image..."):
                        # Load image
                        image = Image.open(uploaded_file)

                        # Convert to RGB if necessary
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        # Get image data
                        img_array = np.array(image)
                        original_shape = img_array.shape

                        # Flatten image data
                        img_bytes = img_array.tobytes()

                        # Generate random IV
                        iv = os.urandom(16)

                        # Encrypt
                        key_bytes = pad_key(encrypt_key)
                        aes = AES(key_bytes, sbox)
                        encrypted_data = aes.encrypt_cbc(img_bytes, iv)

                        # Store metadata and encrypted data
                        metadata = {
                            "shape": original_shape,
                            "iv": iv,
                            "encrypted": encrypted_data,
                        }

                        st.session_state.encrypted_image = metadata
                        st.session_state.img_enc_key_used = encrypt_key

                        # Create encrypted image visualization (noise)
                        encrypted_array = np.frombuffer(
                            encrypted_data[: original_shape[0] * original_shape[1] * 3],
                            dtype=np.uint8,
                        ).reshape(original_shape)
                        encrypted_img = Image.fromarray(encrypted_array, "RGB")

                        st.success("✅ Image encrypted successfully!")
                        st.image(
                            encrypted_img,
                            caption="Encrypted Image (Visual)",
                            width="stretch",
                        )

                        # Stats
                        col_a, col_b = st.columns(2)
                        col_a.metric("Original Size", f"{len(img_bytes)} bytes")
                        col_b.metric("Encrypted Size", f"{len(encrypted_data)} bytes")

                        # Download button
                        output = io.BytesIO()
                        np.savez_compressed(
                            output, shape=original_shape, iv=iv, data=encrypted_data
                        )
                        output.seek(0)

                        st.download_button(
                            label="💾 Download Encrypted Image Data",
                            data=output,
                            file_name="encrypted_image.npz",
                            mime="application/octet-stream",
                        )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # DECRYPT COLUMN
    with col2:
        st.subheader("🔓 Decrypt Image")

        # Key input
        decrypt_key = st.text_input(
            "Decryption Key (any length):",
            value=st.session_state.get("img_enc_key_used", "MySecretKey12345"),
            key="img_dec_key",
        )

        # Show padded key length
        key_info = f"Key length: {len(decrypt_key)} chars"
        if len(decrypt_key) < 16:
            key_info += f" → will be padded to 16 bytes"
        elif len(decrypt_key) > 16:
            key_info += f" → will be truncated to 16 bytes"
        else:
            key_info += " ✓ (exactly 16 bytes)"
        st.caption(key_info)

        # File upload or use last encrypted
        decrypt_option = st.radio(
            "Choose source:",
            ["Use last encrypted image", "Upload encrypted file"],
            key="decrypt_source",
        )

        encrypted_upload = None
        if decrypt_option == "Upload encrypted file":
            encrypted_upload = st.file_uploader(
                "Upload encrypted image data (.npz):",
                type=["npz"],
                key="enc_img_upload",
            )

        # Decrypt button
        if st.button("🔓 Decrypt Image", type="primary", width="stretch"):
            if not decrypt_key:
                st.error("❌ Please enter a decryption key")
            else:
                try:
                    with st.spinner("Decrypting image..."):
                        # Get encrypted data
                        if decrypt_option == "Use last encrypted image":
                            if not hasattr(st.session_state, "encrypted_image"):
                                st.error(
                                    "❌ No encrypted image available. Encrypt an image first."
                                )
                                st.stop()
                            metadata = st.session_state.encrypted_image
                            original_shape = metadata["shape"]
                            iv = metadata["iv"]
                            encrypted_data = metadata["encrypted"]
                        else:
                            if not encrypted_upload:
                                st.error("❌ Please upload an encrypted file")
                                st.stop()

                            # Load from file
                            data = np.load(encrypted_upload)
                            original_shape = tuple(data["shape"])
                            iv = bytes(data["iv"])
                            encrypted_data = bytes(data["data"])

                        # Decrypt
                        key_bytes = pad_key(decrypt_key)
                        aes = AES(key_bytes, sbox)
                        decrypted_data = aes.decrypt_cbc(encrypted_data, iv)

                        # Reconstruct image
                        img_array = np.frombuffer(
                            decrypted_data[: original_shape[0] * original_shape[1] * 3],
                            dtype=np.uint8,
                        ).reshape(original_shape)

                        decrypted_img = Image.fromarray(img_array, "RGB")

                        st.success("✅ Image decrypted successfully!")
                        st.image(
                            decrypted_img,
                            caption="Decrypted Image",
                            width="stretch",
                        )

                        # Download button
                        output = io.BytesIO()
                        decrypted_img.save(output, format="PNG")
                        output.seek(0)

                        st.download_button(
                            label="💾 Download Decrypted Image",
                            data=output,
                            file_name="decrypted_image.png",
                            mime="image/png",
                        )

                except Exception as e:
                    st.error(f"❌ Decryption error: {str(e)}")
                    st.info(
                        "Make sure you're using the correct key and S-box that were used for encryption."
                    )

    # Help section
    st.write("---")

    with st.expander("ℹ️ How to use Image Encryption"):
        st.markdown(
            """
        ### 📝 Instructions
        
        **To Encrypt:**
        1. Select an S-box (standard or custom)
        2. Enter an encryption key (any length - will be padded/truncated to 16 bytes)
        3. Upload an image (PNG, JPG, BMP)
        4. Click "Encrypt Image"
        5. Download the encrypted data file (.npz) to save it
        
        **To Decrypt:**
        1. Use the SAME S-box and key that was used for encryption
        2. Either use the last encrypted image or upload an encrypted .npz file
        3. Click "Decrypt Image"
        4. Download the decrypted image if desired
        
        **Key Handling:**
        - Keys shorter than 16 bytes are padded with zeros
        - Keys longer than 16 bytes are truncated to 16 bytes
        - For best security, use exactly 16 characters
        - Example keys: "secret", "MySecretKey12345", "a very long password"
        
        **Security Notes:**
        - The encrypted image appears as random noise
        - The entire pixel data is encrypted using AES-CBC
        - Image dimensions and format are preserved
        - The .npz file contains the encrypted data, IV, and shape information
        - Anyone with the key and S-box can decrypt the image
        
        **Testing Custom S-boxes:**
        - This is perfect for testing your custom S-boxes from the paper
        - Compare encryption quality using histogram analysis
        - The encrypted image should look completely random
        
        **Performance:**
        - Larger images take longer to encrypt/decrypt
        - Typical time: < 5 seconds for 512×512 images
        """
        )


def calculate_entropy(image_data):
    """
    Calculate entropy of image data.
    H(m) = Σ P(m_i) * log2(1/P(m_i))

    Ideal value: 8.0 for perfect randomness
    """
    # Flatten if multi-channel
    if len(image_data.shape) == 3:
        data = image_data.flatten()
    else:
        data = image_data.flatten()

    # Calculate histogram
    hist, _ = np.histogram(data, bins=256, range=(0, 256))

    # Calculate probabilities
    probs = hist / float(np.sum(hist))

    # Remove zero probabilities
    probs = probs[probs > 0]

    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs))

    return entropy


def calculate_npcr(image1, image2):
    """
    Calculate Number of Pixels Change Rate (NPCR).
    NPCR = (Σ D(i,j) / (M*N)) * 100%

    Where D(i,j) = 0 if C1(i,j) = C2(i,j), else 1

    Ideal value: 99.6094% for perfect sensitivity
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Compare pixels
    diff = (image1 != image2).astype(int)

    # Calculate NPCR
    npcr = (np.sum(diff) / diff.size) * 100.0

    return npcr


def calculate_uaci(image1, image2):
    """
    Calculate Unified Average Changing Intensity (UACI).
    UACI = (1/(M*N)) * Σ |C1(i,j) - C2(i,j)| / 255 * 100%

    Ideal value: 33.4635% for 8-bit images
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate absolute difference
    diff = np.abs(image1.astype(float) - image2.astype(float))

    # Calculate UACI
    uaci = (np.sum(diff) / (diff.size * 255.0)) * 100.0

    return uaci


def plot_histogram(image_data, title="Histogram"):
    """
    Plot histogram of image data.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    if len(image_data.shape) == 3:
        # Color image - plot R, G, B separately
        colors = ["red", "green", "blue"]
        for i, color in enumerate(colors):
            hist, bins = np.histogram(
                image_data[:, :, i].flatten(), bins=256, range=(0, 256)
            )
            ax.plot(bins[:-1], hist, color=color, alpha=0.7, label=color.upper())
        ax.legend()
    else:
        # Grayscale image
        hist, bins = np.histogram(image_data.flatten(), bins=256, range=(0, 256))
        ax.bar(bins[:-1], hist, width=1, color="gray", alpha=0.7)

    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    return fig


def render_image_analysis():
    """
    Image encryption analysis interface based on the paper.
    Implements: Histogram Analysis, Entropy Test, NPCR Test
    """
    st.info(
        """
    **Image Encryption Analysis**
    
    Test encrypted images using metrics from the paper:
    - **Histogram Analysis**: Visual distribution of pixel values
    - **Entropy Test**: Measure randomness (ideal: 8.0)
    - **NPCR Test**: Sensitivity to key/plaintext changes (ideal: 99.6%)
    - **UACI Test**: Average intensity change (ideal: 33.4%)
    """
    )

    # Analysis mode selection
    analysis_mode = st.radio(
        "Select analysis mode:",
        ["Single Image Analysis", "Compare Two Images (NPCR/UACI)"],
        horizontal=True,
    )

    if analysis_mode == "Single Image Analysis":
        st.subheader("📊 Single Image Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Image**")
            original_file = st.file_uploader(
                "Upload original image:",
                type=["png", "jpg", "jpeg", "bmp"],
                key="analysis_original",
            )

            if original_file:
                original_img = Image.open(original_file)
                if original_img.mode != "RGB":
                    original_img = original_img.convert("RGB")
                st.image(original_img, caption="Original", width="stretch")

        with col2:
            st.write("**Encrypted Image**")
            encrypted_file = st.file_uploader(
                "Upload encrypted image (.npz):",
                type=["npz"],
                key="analysis_encrypted",
            )

            if encrypted_file:
                data = np.load(encrypted_file)
                encrypted_data = bytes(data["data"])
                shape = tuple(data["shape"])
                encrypted_array = np.frombuffer(
                    encrypted_data[: shape[0] * shape[1] * 3],
                    dtype=np.uint8,
                ).reshape(shape)
                encrypted_img = Image.fromarray(encrypted_array, "RGB")
                st.image(encrypted_img, caption="Encrypted", width="stretch")

        if st.button("🔬 Analyze Images", type="primary", width="stretch"):
            if not original_file or not encrypted_file:
                st.error("❌ Please upload both original and encrypted images")
            else:
                with st.spinner("Analyzing images..."):
                    # Convert to numpy arrays
                    original_array = np.array(original_img)
                    # encrypted_array already loaded from .npz above

                    # Check dimensions match
                    if original_array.shape != encrypted_array.shape:
                        st.error("❌ Images must have the same dimensions!")
                        st.stop()

                    # Calculate entropy
                    original_entropy = calculate_entropy(original_array)
                    encrypted_entropy = calculate_entropy(encrypted_array)

                    # Results
                    st.write("---")
                    st.subheader("📈 Analysis Results")

                    # Entropy metrics
                    st.write("**Entropy Analysis**")
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Original Entropy", f"{original_entropy:.4f}")
                    col2.metric("Encrypted Entropy", f"{encrypted_entropy:.4f}")
                    col3.metric("Ideal", "8.0000")

                    # Entropy assessment
                    if encrypted_entropy >= 7.999:
                        st.success("✅ Excellent entropy! Near perfect randomness.")
                    elif encrypted_entropy >= 7.990:
                        st.success("✅ Very good entropy! High randomness.")
                    elif encrypted_entropy >= 7.950:
                        st.info("ℹ️ Good entropy. Acceptable randomness.")
                    else:
                        st.warning("⚠️ Low entropy. May be predictable.")

                    # Entropy per channel (for color images)
                    if len(encrypted_array.shape) == 3:
                        st.write("**Entropy per Channel:**")
                        col1, col2, col3 = st.columns(3)

                        r_entropy = calculate_entropy(encrypted_array[:, :, 0])
                        g_entropy = calculate_entropy(encrypted_array[:, :, 1])
                        b_entropy = calculate_entropy(encrypted_array[:, :, 2])

                        col1.metric("Red Channel", f"{r_entropy:.4f}")
                        col2.metric("Green Channel", f"{g_entropy:.4f}")
                        col3.metric("Blue Channel", f"{b_entropy:.4f}")

                    # Histograms
                    st.write("---")
                    st.write("**Histogram Analysis**")

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_orig = plot_histogram(
                            original_array, "Original Image Histogram"
                        )
                        st.pyplot(fig_orig)
                        plt.close()
                        st.caption("Original image shows non-uniform distribution")

                    with col2:
                        fig_enc = plot_histogram(
                            encrypted_array, "Encrypted Image Histogram"
                        )
                        st.pyplot(fig_enc)
                        plt.close()

                        # Check histogram uniformity
                        if len(encrypted_array.shape) == 3:
                            flat_data = encrypted_array.flatten()
                        else:
                            flat_data = encrypted_array.flatten()

                        hist, _ = np.histogram(flat_data, bins=256, range=(0, 256))
                        hist_std = np.std(hist)
                        hist_mean = np.mean(hist)

                        if hist_std / hist_mean < 0.1:
                            st.caption("✅ Encrypted histogram is very uniform (flat)")
                        elif hist_std / hist_mean < 0.2:
                            st.caption("✅ Encrypted histogram is reasonably uniform")
                        else:
                            st.caption("⚠️ Encrypted histogram shows some patterns")

                    # Comparison with paper results
                    st.write("---")
                    st.write("**Comparison with Paper Results**")

                    comparison_data = {
                        "Method": [
                            "Your Result",
                            "Paper (woman_blonde)",
                            "Paper (Best)",
                            "Ideal",
                        ],
                        "Entropy": [
                            f"{encrypted_entropy:.4f}",
                            "7.9993",
                            "7.9994",
                            "8.0000",
                        ],
                        "Status": [
                            "✅" if encrypted_entropy >= 7.999 else "⚠️",
                            "✅",
                            "✅",
                            "✅",
                        ],
                    }

                    st.dataframe(
                        pd.DataFrame(comparison_data),
                        width="stretch",
                        hide_index=True,
                    )

    else:  # Compare Two Images
        st.subheader("⚖️ Compare Two Images (NPCR/UACI Test)")

        st.info(
            """
        **NPCR Test**: Measures sensitivity to key/plaintext changes.
        Upload two encrypted versions of the same image (with 1-bit key difference) to test differential attack resistance.
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Encrypted Image 1**")
            image1_file = st.file_uploader(
                "Upload first encrypted image (.npz):",
                type=["npz"],
                key="npcr_image1",
            )

            if image1_file:
                data1 = np.load(image1_file)
                encrypted_data1 = bytes(data1["data"])
                shape1 = tuple(data1["shape"])
                array1 = np.frombuffer(
                    encrypted_data1[: shape1[0] * shape1[1] * 3],
                    dtype=np.uint8,
                ).reshape(shape1)
                img1 = Image.fromarray(array1, "RGB")
                st.image(img1, caption="Encrypted Image 1", width="stretch")

        with col2:
            st.write("**Encrypted Image 2**")
            image2_file = st.file_uploader(
                "Upload second encrypted image (.npz):",
                type=["npz"],
                key="npcr_image2",
            )

            if image2_file:
                data2 = np.load(image2_file)
                encrypted_data2 = bytes(data2["data"])
                shape2 = tuple(data2["shape"])
                array2 = np.frombuffer(
                    encrypted_data2[: shape2[0] * shape2[1] * 3],
                    dtype=np.uint8,
                ).reshape(shape2)
                img2 = Image.fromarray(array2, "RGB")
                st.image(img2, caption="Encrypted Image 2", width="stretch")

        if st.button("📊 Calculate NPCR & UACI", type="primary", width="stretch"):
            if not image1_file or not image2_file:
                st.error("❌ Please upload both encrypted images")
            else:
                with st.spinner("Calculating metrics..."):
                    # Arrays already loaded from .npz files above

                    # Check dimensions match
                    if array1.shape != array2.shape:
                        st.error("❌ Images must have the same dimensions!")
                        st.stop()

                    # Calculate NPCR and UACI
                    npcr = calculate_npcr(array1, array2)
                    uaci = calculate_uaci(array1, array2)

                    # Results
                    st.write("---")
                    st.subheader("📊 Differential Analysis Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("NPCR", f"{npcr:.4f}%")
                    col2.metric("UACI", f"{uaci:.4f}%")
                    col3.metric("Image Size", f"{array1.shape[0]}×{array1.shape[1]}")

                    # NPCR assessment
                    st.write("**NPCR Analysis:**")
                    if npcr >= 99.60:
                        st.success(
                            "✅ Excellent NPCR! Strong resistance to differential attacks."
                        )
                    elif npcr >= 99.50:
                        st.info("ℹ️ Good NPCR. Acceptable differential resistance.")
                    else:
                        st.warning(
                            "⚠️ Low NPCR. May be vulnerable to differential attacks."
                        )

                    # UACI assessment
                    st.write("**UACI Analysis:**")
                    uaci_dev = abs(uaci - 33.4635)
                    if uaci_dev <= 0.5:
                        st.success(f"✅ Excellent UACI! Deviation: {uaci_dev:.4f}%")
                    elif uaci_dev <= 1.0:
                        st.info(f"ℹ️ Good UACI. Deviation: {uaci_dev:.4f}%")
                    else:
                        st.warning(f"⚠️ UACI deviation: {uaci_dev:.4f}%")

                    # Per-channel analysis for color images
                    if len(array1.shape) == 3:
                        st.write("---")
                        st.write("**Per-Channel Analysis:**")

                        channels = ["Red", "Green", "Blue"]
                        for i, channel in enumerate(channels):
                            npcr_ch = calculate_npcr(array1[:, :, i], array2[:, :, i])
                            uaci_ch = calculate_uaci(array1[:, :, i], array2[:, :, i])

                            col1, col2 = st.columns(2)
                            col1.metric(f"{channel} NPCR", f"{npcr_ch:.4f}%")
                            col2.metric(f"{channel} UACI", f"{uaci_ch:.4f}%")

                    # Comparison with paper
                    st.write("---")
                    st.write("**Comparison with Paper Results**")

                    comparison_data = {
                        "Method": [
                            "Your Result",
                            "Paper (woman_blonde)",
                            "Paper (Best)",
                            "Ideal",
                        ],
                        "NPCR (%)": [f"{npcr:.4f}", "99.6288", "99.6288", "99.6094"],
                        "UACI (%)": [f"{uaci:.4f}", "33.4635", "33.4635", "33.4635"],
                        "Status": ["✅" if npcr >= 99.60 else "⚠️", "✅", "✅", "✅"],
                    }

                    st.dataframe(
                        pd.DataFrame(comparison_data),
                        width="stretch",
                        hide_index=True,
                    )

                    # Visual difference
                    st.write("---")
                    st.write("**Visual Difference Map**")

                    diff_map = np.abs(
                        array1.astype(float) - array2.astype(float)
                    ).astype(np.uint8)
                    st.image(
                        diff_map,
                        caption="Difference Map (lighter = more difference)",
                        width="stretch",
                    )

    # Help section
    st.write("---")

    with st.expander("ℹ️ How to use Image Analysis"):
        st.markdown(
            """
        ### 📝 Single Image Analysis
        
        **Purpose**: Evaluate encryption quality of a single encrypted image.
        
        **Steps**:
        1. Upload the original (plaintext) image
        2. Upload the encrypted (ciphertext) image
        3. Click "Analyze Images"
        4. Review entropy and histogram results
        
        **What to look for**:
        - **Entropy near 8.0**: Indicates good randomness
        - **Flat histogram**: Shows uniform pixel distribution
        - **Comparison with paper**: Your results vs published results
        
        ---
        
        ### ⚖️ NPCR/UACI Test
        
        **Purpose**: Test sensitivity to key or plaintext changes (differential attack resistance).
        
        **Steps**:
        1. Encrypt the same image twice with keys differing by 1 bit
        2. Upload both encrypted images
        3. Click "Calculate NPCR & UACI"
        4. Review sensitivity metrics
        
        **What to look for**:
        - **NPCR ≥ 99.60%**: Excellent differential sensitivity
        - **UACI ≈ 33.46%**: Ideal average intensity change
        - **Per-channel consistency**: All channels should have similar values
        
        ---
        
        ### 📊 Metrics Explained
        
        **Entropy (H)**:
        - Measures randomness: H = Σ P(m_i) × log₂(1/P(m_i))
        - Range: 0 to 8 (for 8-bit images)
        - Ideal: 8.0 (perfect randomness)
        - Paper results: 7.9994
        
        **NPCR (Number of Pixels Change Rate)**:
        - Measures % of pixels that change
        - Formula: (Σ D(i,j) / (M×N)) × 100%
        - Ideal: 99.6094%
        - Paper results: 99.6288%
        
        **UACI (Unified Average Changing Intensity)**:
        - Measures average intensity change
        - Formula: (1/(M×N)) × Σ |C1-C2|/255 × 100%
        - Ideal: 33.4635%
        - Paper results: 33.4635%
        
        ---
        
        ### 🎯 Testing Your Custom S-boxes
        
        1. Go to "Image Encryption" tab
        2. Select your custom S-box
        3. Encrypt an image
        4. Download the encrypted image
        5. Come to this tab to analyze it
        6. Compare your results with paper benchmarks
        
        **Pro tip**: Test with standard images (Lena, Cameraman) to compare with literature!
        """
        )


# Main execution
if __name__ == "__main__":
    st.set_page_config(page_title="AES Implementation", page_icon="🔐", layout="wide")

    render_aes_implementation()
