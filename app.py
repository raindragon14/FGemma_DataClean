import streamlit as st
import pandas as pd
import json
import re
import os
from io import BytesIO

# Import backend yang sudah kita buat
from backend_executor import DataCleaningExecutor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Data Cleaner (FunctionGemma)",
    page_icon="🧹",
    layout="wide"
)

# --- CLASS: LLM BRIDGE (JEMBATAN KE MODEL AI) ---
class LLMProcessor:
    """
    Kelas ini menangani komunikasi dengan Model AI (FunctionGemma/DeepSeek/dll)
    dan memparsing outputnya menjadi format yang dimengerti backend.
    """
    
    def __init__(self, api_key=None, provider="simulation"):
        self.api_key = api_key
        self.provider = provider

    def get_function_call(self, user_query, df_columns):
        """
        Mengirim prompt ke LLM dan mendapatkan string function call.
        """
        system_prompt = f"""
        You are an expert Data Cleaning Assistant using FunctionGemma tools.
        Available Columns: {list(df_columns)}
        """

        # --- OPSI 1: SIMULASI (Untuk Testing UI tanpa Model) ---
        if self.provider == "simulation":
            # Hardcoded logic untuk demonstrasi UI
            query_lower = user_query.lower()
            if "hapus" in query_lower and "kosong" in query_lower:
                col = [c for c in df_columns if c.lower() in query_lower]
                target_col = col[0] if col else df_columns[0]
                return f'<start_function_call>call:handle_missing_and_nulls{{"column_name": "{target_col}", "strategy": "drop_row"}}<end_function_call>'
            
            elif "kapital" in query_lower or "besar" in query_lower:
                col = [c for c in df_columns if c.lower() in query_lower]
                target_col = col[0] if col else df_columns[0]
                return f'<start_function_call>call:clean_text_normalization{{"column_name": "{target_col}", "operations": ["to_upper", "trim_whitespace"]}}<end_function_call>'
            
            return None

        # --- OPSI 2: REAL API (DeepSeek/OpenAI/Local FunctionGemma) ---
        # Nanti Anda ganti bagian ini dengan kode inference model FunctionGemma 270M Anda
        # contoh: model.generate(prompt)
        # return output_model_string
        
        return None

    def parse_output(self, llm_output):
        """
        Memecah string output model menjadi nama fungsi dan argumen JSON.
        Format: <start_function_call>call:func_name{args}<end_function_call>
        """
        if not llm_output:
            return None, None, "Tidak ada output dari model."

        # Regex untuk menangkap call:...{...}
        # Pattern ini cocok dengan format FunctionGemma
        pattern = r"call:(\w+)(\{.*\})"
        match = re.search(pattern, llm_output)

        if match:
            tool_name = match.group(1)
            json_args = match.group(2)
            try:
                args = json.loads(json_args)
                return tool_name, args, None
            except json.JSONDecodeError:
                return None, None, "Gagal parsing JSON argumen."
        else:
            return None, None, "Format function call tidak dikenali."

# --- MAIN APP UI ---

def main():
    st.title("🧹 AI Data Cleaner Agent")
    st.markdown("Powered by **FunctionGemma 270M** (Fine-Tuned)")

    # Sidebar: Setup
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        # Nanti bisa diganti "Local Model" atau "API"
        mode = st.selectbox("Mode Model", ["Simulasi (Test UI)", "DeepSeek API", "Local FunctionGemma"])
        api_key = ""
        if mode == "DeepSeek API":
            api_key = st.text_input("API Key", type="password")
        
        st.divider()
        st.info("Upload file Excel/CSV, lalu ketik instruksi dalam Bahasa Indonesia.")

    # 1. State Management (Agar data tidak reset saat klik tombol)
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 2. File Uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

    if uploaded_file:
        # Load data hanya sekali
        if st.session_state.df is None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f"File '{uploaded_file.name}' berhasil dimuat!")
            except Exception as e:
                st.error(f"Gagal memuat file: {e}")

    # Tampilkan Main Interface jika data ada
    if st.session_state.df is not None:
        
        # Layout: Kiri (Data), Kanan (Chat/Kontrol)
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state.df, use_container_width=True, height=400)
            
            # Statistik Singkat
            st.caption(f"Rows: {st.session_state.df.shape[0]} | Columns: {st.session_state.df.shape[1]}")

        with col2:
            st.subheader("💬 AI Command")
            user_query = st.text_area("Instruksi", placeholder="Contoh: Tolong hapus baris yang kolom 'Usia' nya kosong...", height=100)
            
            process_btn = st.button("🚀 Jalankan", type="primary", use_container_width=True)

            if process_btn and user_query:
                with st.spinner("Sedang berpikir..."):
                    # 1. Init Processor
                    processor = LLMProcessor(provider="simulation" if mode == "Simulasi (Test UI)" else "api")
                    
                    # 2. Get LLM Response
                    # (Di sini kita kirim nama kolom agar model tau konteks)
                    raw_response = processor.get_function_call(user_query, st.session_state.df.columns)
                    
                    # 3. Parse Response
                    tool_name, args, error = processor.parse_output(raw_response)

                    if error:
                        st.error(f"Error AI: {error}")
                        if raw_response: st.code(raw_response) # Debugging info
                    else:
                        # 4. Execute Backend
                        executor = DataCleaningExecutor(st.session_state.df)
                        
                        # Tampilkan apa yang akan dijalankan
                        st.info(f"🛠 **Tool:** `{tool_name}`")
                        st.json(args, expanded=False)

                        # Eksekusi
                        new_df = executor.execute_tool(tool_name, args)
                        
                        # Update State
                        st.session_state.df = new_df
                        st.session_state.history.append(f"✅ {tool_name}: {user_query}")
                        st.rerun() # Refresh halaman untuk update tabel

            st.divider()
            st.subheader("Riwayat Perubahan")
            for log in reversed(st.session_state.history):
                st.text(log)

        # Download Section
        st.divider()
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Hasil",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
