import streamlit as st
import pandas as pd
import json
import re
import os
import logging
from llama_cpp import Llama

# Import Backend Executor
from backend import DataCleaningExecutor

# --- KONFIGURASI ---
# Ganti dengan path ke model GGUF hasil fine-tuning Anda
MODEL_PATH = "models/functiongemma-270m-it-finetuned.gguf" 
# Context window (270M cukup kecil, 2048 atau 4096 aman untuk laptop)
N_CTX = 4096 

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AppInterface")

st.set_page_config(
    page_title="AI Data Cleaner (Local CPU)",
    page_icon="🧹",
    layout="wide"
)

# --- DEFINISI TOOLS (Harus sama persis dengan saat training) ---
# (Disarankan dipindah ke file config.py terpisah agar rapi, tapi ditaruh sini agar standalone)
TOOLS_SCHEMA = [
  {
    "name": "clean_text_normalization",
    "description": "Membersihkan teks input: kapitalisasi, spasi, typo, hapus gelar, dan regex.",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string", "description": "Nama kolom target."},
        "operations": {"type": "array", "items": {"type": "string", "enum": ["trim_whitespace", "to_lower", "to_upper", "to_title_case", "remove_special_chars", "normalize_whitespace"]}, "description": "Urutan operasi pembersihan."},
        "regex_replacements": {"type": "array", "items": {"type": "object", "properties": {"pattern": {"type": "string"}, "replacement": {"type": "string"}}, "required": ["pattern", "replacement"]}, "description": "List aturan regex kustom."}
      },
      "required": ["column_name", "operations"]
    }
  },
  {
    "name": "parse_indonesian_numeric_currency",
    "description": "Parsing angka/uang format Indonesia (Rp, titik ribuan, koma desimal, singkatan 'jt'/'rb').",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string", "description": "Nama kolom target."},
        "decimal_separator": {"type": "string", "default": ",", "description": "Pemisah desimal."},
        "thousand_separator": {"type": "string", "default": ".", "description": "Pemisah ribuan."},
        "currency_symbol_cleanup": {"type": "boolean", "description": "Hapus simbol mata uang."},
        "handle_abbreviations": {"type": "boolean", "description": "Konversi 'jt', 'm', 'rb'."}
      },
      "required": ["column_name"]
    }
  },
  {
    "name": "standardize_indonesian_datetime",
    "description": "Parsing tanggal lokal (Indonesia) ke format ISO standar.",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string"},
        "input_formats": {"type": "array", "items": {"type": "string"}},
        "target_format": {"type": "string", "default": "%Y-%m-%d"},
        "timezone_conversion": {"type": "object", "properties": {"from_tz": {"type": "string"}, "to_tz": {"type": "string"}}, "required": ["from_tz", "to_tz"]}
      },
      "required": ["column_name"]
    }
  },
  {
    "name": "handle_missing_and_nulls",
    "description": "Menangani data kosong.",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string"},
        "missing_indicators": {"type": "array", "items": {"type": "string"}, "default": ["", "NaN", "NULL", "-", "n/a"]},
        "strategy": {"type": "string", "enum": ["drop_row", "fill_value", "fill_mean", "fill_median", "fill_mode", "forward_fill", "backward_fill"]},
        "fill_value": {"type": "string"}
      },
      "required": ["column_name", "strategy"]
    }
  },
  {
    "name": "normalize_categorical_values",
    "description": "Mapping kategori.",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string"},
        "mapping_rules": {"type": "object"},
        "case_insensitive": {"type": "boolean", "default": True},
        "default_value": {"type": "string"}
      },
      "required": ["column_name", "mapping_rules"]
    }
  },
  {
    "name": "deduplicate_and_fuzzy_match",
    "description": "Hapus duplikasi data.",
    "parameters": {
      "type": "object",
      "properties": {
        "subset_columns": {"type": "array", "items": {"type": "string"}},
        "method": {"type": "string", "enum": ["exact", "fuzzy"]},
        "keep": {"type": "string", "enum": ["first", "last", "none"], "default": "first"},
        "fuzzy_threshold": {"type": "number"}
      },
      "required": ["subset_columns", "method"]
    }
  },
  {
    "name": "validate_and_filter_outliers",
    "description": "Validasi dan filter data.",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string"},
        "validation_type": {"type": "string", "enum": ["regex_match", "numeric_range", "length_check", "list_membership"]},
        "criteria": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}, "pattern": {"type": "string"}, "allowed_values": {"type": "array", "items": {"type": "string"}}}},
        "action": {"type": "string", "enum": ["drop", "nullify", "flag"]}
      },
      "required": ["column_name", "validation_type", "action"]
    }
  },
  {
    "name": "split_and_extract_entities",
    "description": "Memecah kolom atau ekstrak entitas.",
    "parameters": {
      "type": "object",
      "properties": {
        "column_name": {"type": "string"},
        "operation": {"type": "string", "enum": ["split_by_delimiter", "extract_regex", "extract_json_field"]},
        "delimiter": {"type": "string"},
        "regex_pattern": {"type": "string"},
        "new_column_names": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["column_name", "operation", "new_column_names"]
    }
  }
]

# --- LOAD MODEL (CACHED) ---
import multiprocessing

@st.cache_resource
def load_llm():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file tidak ditemukan di: {MODEL_PATH}")
        st.stop()
    
    st.toast("Memuat model AI ke Memori...", icon="🧠")
    
    # HITUNG THREADS OPTIMAL
    # Gunakan physical cores jika memungkinkan, sisakan 2 untuk System/UI
    total_cores = multiprocessing.cpu_count()
    optimal_threads = max(1, total_cores - 2) 

    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,          # Context window aman untuk laptop
            n_batch=1024,        # [OPTIMASI] Mempercepat pembacaan prompt panjang (schema)
            n_threads=optimal_threads, # [OPTIMASI] Agar UI tidak freeze
            n_gpu_layers=0,      # Force CPU
            verbose=False,       # Matikan log spam di terminal
            use_mlock=True       # [OPTIMASI] Kunci model di RAM agar tidak swap ke disk (lebih stabil)
        )
        return llm
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# --- LLM INFERENCE ENGINE ---
class LocalLLMProcessor:
    def __init__(self, llm):
        self.llm = llm

    def generate_function_call(self, user_query, df_columns):
        """
        Membangun prompt FunctionGemma dan melakukan inferensi.
        """
        # Format Prompt FunctionGemma (Gemma 3 template)
        # Template ini krusial agar model sadar tools
        system_content = f"You are an expert Data Cleaning Assistant. available_columns: {list(df_columns)}. Tools:\n{json.dumps(TOOLS_SCHEMA)}"
        
        prompt = f"""<start_of_turn>developer
{system_content}<end_of_turn>
<start_of_turn>user
{user_query}<end_of_turn>
<start_of_turn>model
"""
        
        # Inference parameter (Optimasi untuk CPU: temperature rendah biar cepat & pasti)
        output = self.llm(
            prompt,
            max_tokens=1024,      # [FIX] Dinaikkan dari 512 untuk mencegah JSON terpotong
            stop=["<end_function_call>", "<end_of_turn>"],
            temperature=1.0,      # [FIX] Greedy decoding untuk format JSON yang strict
            top_p=0.95,            # [FIX] Tidak perlu sampling acak
            top_k=64,             # Standard filtering
            repeat_penalty=1.1,   # [FIX] Mencegah looping jika model bingung (cth: }}}}}})
            echo=False
        )
        
        generated_text = output['choices'][0]['text']
        
        # Safety net: Jika model lupa menutup tag (jarang terjadi di temp 0, tapi preventif)
        if "<start_function_call>" in generated_text and "<end_function_call>" not in generated_text:
            generated_text += "<end_function_call>"
            
        return generated_text.strip()

    def parse_output(self, llm_output):
        """
        Parsing output raw model menjadi (tool_name, args)
        """
        logger.info(f"Raw LLM Output: {llm_output}")
        
        # Regex untuk menangkap pattern: <start_function_call>call:nama_tool{json_args}<end_function_call>
        # Kita handle kalau-kalau model lupa tag penutup atau pembuka
        clean_output = llm_output.replace("<start_function_call>", "").replace("<end_function_call>", "").strip()
        
        pattern = r"call:(\w+)(\{.*\})"
        match = re.search(pattern, clean_output, re.DOTALL)

        if match:
            tool_name = match.group(1)
            json_args_str = match.group(2)
            try:
                args = json.loads(json_args_str)
                return tool_name, args, None
            except json.JSONDecodeError:
                return None, None, f"Gagal parsing JSON argumen: {json_args_str}"
        else:
            return None, None, "Model tidak menghasilkan function call yang valid."

# --- MAIN APP ---
def main():
    st.title("💻 Local AI Data Cleaner")
    st.caption("Running on CPU - FunctionGemma 270M")

    # Load Model
    llm = load_llm()
    processor = LocalLLMProcessor(llm)

    # State Init
    if 'df' not in st.session_state: st.session_state.df = None
    if 'history' not in st.session_state: st.session_state.history = []

    # Sidebar
    with st.sidebar:
        st.header("📂 Dataset")
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file and st.session_state.df is None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        if st.button("Reset Data"):
            st.session_state.df = None
            st.session_state.history = []
            st.rerun()

    # Main Interface
    if st.session_state.df is not None:
        col_data, col_chat = st.columns([2, 1])

        with col_data:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.df, height=500, use_container_width=True)

        with col_chat:
            st.subheader("🤖 Instruksi AI")
            user_input = st.text_area("Perintah (Bahasa Indonesia)", height=100, 
                                    placeholder="Contoh: Normalisasi kolom Kota jadi huruf kapital semua")
            
            if st.button("Proses", type="primary", use_container_width=True):
                if not user_input:
                    st.warning("Masukkan perintah dulu.")
                else:
                    with st.spinner("Model sedang berpikir (CPU)..."):
                        # 1. Generate
                        raw_output = processor.generate_function_call(user_input, st.session_state.df.columns)
                        
                        # 2. Parse
                        tool_name, args, error = processor.parse_output(raw_output)

                        if error:
                            st.error(f"Gagal: {error}")
                            with st.expander("Debug Info"):
                                st.code(raw_output)
                        else:
                            # 3. Execute Backend
                            st.success(f"Menjalankan: {tool_name}")
                            st.json(args, expanded=False)
                            
                            executor = DataCleaningExecutor(st.session_state.df)
                            new_df = executor.execute_tool(tool_name, args)
                            
                            st.session_state.df = new_df
                            st.session_state.history.append(f"✅ {tool_name}: {user_input}")
                            st.rerun()

            st.divider()
            st.write("Riwayat:")
            for h in reversed(st.session_state.history):
                st.caption(h)
            
            # Download
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil CSV", csv, "cleaned_data.csv", "text/csv")

    else:
        st.info("Silakan upload file di sidebar untuk memulai.")

if __name__ == "__main__":
    main()
