"""
AES Implementation with Custom S-box
Complete encrypt/decrypt interface for text and images using custom or standard S-boxes
"""

import io
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


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
    tab1, tab2 = st.tabs(["📝 Text Encryption", "🖼️ Image Encryption"])

    with tab1:
        render_text_encryption()

    with tab2:
        render_image_encryption()


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
                            decrypted_img, caption="Decrypted Image", width="stretch"
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


# Main execution
if __name__ == "__main__":
    st.set_page_config(page_title="AES Implementation", page_icon="🔐", layout="wide")

    render_aes_implementation()
