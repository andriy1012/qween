# ------------------------------------------------------------
# 1️⃣  IMPORT & KONFIGURASI
# ------------------------------------------------------------
import os
import json
import asyncio
import httpx
import streamlit as st

# ----------  Pengaturan API ----------
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL    = "qwen/qwen2.5-coder-32b-instruct"

# Ambil API‑Key dari environment variable (lebih aman)
API_KEY = os.getenv("NVIDIA_API_KEY") or "YOUR_API_KEY"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ------------------------------------------------------------
# 2️⃣  SETUP STREAMLIT
# ------------------------------------------------------------
st.set_page_config(page_title="Qwen‑Coder Chat (Streamlit)", page_icon="🤖")
st.title("🤖 Qwen 2.5‑Coder Chat – Demo Streamlit")

# -----  Session state untuk riwayat & placeholder -----
if "messages" not in st.session_state:
    # System‑prompt memberi arahan dasar ke model
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "kamu bisa tanya apa saja, "
                "jawaban dengan akurat dan dapat seuai logika"
            ),
        }
    ]

# Placeholder yang menampung seluruh riwayat chat (dibuat sekali)
if "chat_placeholder" not in st.session_state:
    st.session_state.chat_placeholder = st.empty()


# ------------------------------------------------------------
# 3️⃣  FUNGSI RENDER RIWAYAT (bersih, tidak duplikat)
# ------------------------------------------------------------
def render_chat() -> None:
    """Render seluruh riwayat percakapan (kecuali system‑prompt)."""
    placeholder = st.session_state.chat_placeholder
    placeholder.empty()                     # hapus output lama

    rendered_md = ""
    for msg in st.session_state.messages[1:]:   # lewati elemen pertama (system)
        if msg["role"] == "user":
            rendered_md += f"**👤 Anda:** {msg['content']}\n\n"
        else:  # assistant / model
            rendered_md += f"**🤖 Model:** {msg['content']}\n\n"

    placeholder.markdown(rendered_md)


# ------------------------------------------------------------
# 4️⃣  ASYNC CALL KE NVIDIA DENGAN STREAMING
# ------------------------------------------------------------
async def stream_completion(messages):
    """
    Kirim request ke NVIDIA API dengan `stream=True` dan
    yield token‑per‑token.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": True,               # <‑‑ penting!
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
        ) as resp:

            if resp.status_code != 200:
                err_body = await resp.aread()
                raise RuntimeError(
                    f"NVIDIA API error {resp.status_code}: {err_body}"
                )

            # NVIDIA mengirimkan tiap token sebagai baris JSON yang dimulai dengan "data: "
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except Exception:
                        # Jika parsing gagal, lewati baris itu
                        continue


# ------------------------------------------------------------
# 5️⃣  UI INPUT USER
# ------------------------------------------------------------
prompt = st.chat_input("Tulis pertanyaan atau kode …")   # Streamlit ≥1.24
# (Jika versi lama, gunakan st.text_input + st.button)

if prompt:
    # Simpan pertanyaan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder khusus untuk menampilkan token streaming
    streaming_placeholder = st.empty()
    streaming_placeholder.markdown("**🤖 Model:** …")

    # ----------------------------------------------------------------
    # Coroutine yang mengumpulkan token‑per‑token, memperbarui UI,
    # lalu menyimpan jawaban akhir ke riwayat.
    # ----------------------------------------------------------------
    async def generate():
        answer_parts = []
        async for token in stream_completion(st.session_state.messages):
            answer_parts.append(token)
            streaming_placeholder.markdown(
                f"**🤖 Model:** {''.join(answer_parts)}"
            )
        # Simpan jawaban akhir ke riwayat
        st.session_state.messages.append(
            {"role": "assistant", "content": "".join(answer_parts)}
        )
        # Bersihkan placeholder streaming (opsional)
        streaming_placeholder.empty()
        # Render ulang seluruh riwayat (hanya sekali)
        render_chat()

    # Jalankan coroutine
    asyncio.run(generate())
else:
    # Jika belum ada input, tetap render riwayat yang ada
    render_chat()
