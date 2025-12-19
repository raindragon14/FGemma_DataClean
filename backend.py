import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Union, Optional, Any
from dateutil import parser
from datetime import datetime

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("backend_execution.log"), logging.StreamHandler()]
)
logger = logging.getLogger("DataCleanerBackend")

class DataCleaningExecutor:
    """
    Backend engine untuk mengeksekusi function calls dari FunctionGemma.
    Menggunakan Pandas untuk operasi vektorisasi berkinerja tinggi.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> pd.DataFrame:
        """Dispatcher utama untuk memanggil fungsi berdasarkan nama tool."""
        try:
            method = getattr(self, tool_name, None)
            if not method:
                logger.error(f"Tool '{tool_name}' tidak ditemukan/belum diimplementasi.")
                return self.df
            
            logger.info(f"🚀 Menjalankan Tool: {tool_name} | Args: {args}")
            return method(**args)
        
        except Exception as e:
            logger.error(f"CRITICAL ERROR di {tool_name}: {str(e)}", exc_info=True)
            return self.df

    # --- 1. CLEAN TEXT NORMALIZATION ---
    def clean_text_normalization(self, column_name: str, operations: List[str], 
                                 regex_replacements: Optional[List[Dict[str, str]]] = None) -> pd.DataFrame:
        if column_name not in self.df.columns:
            logger.warning(f"Kolom '{column_name}' tidak ditemukan.")
            return self.df

        # Pastikan tipe data string
        series = self.df[column_name].astype(str)

        for op in operations:
            if op == "trim_whitespace":
                series = series.str.strip()
            elif op == "to_lower":
                series = series.str.lower()
            elif op == "to_upper":
                series = series.str.upper()
            elif op == "to_title_case":
                series = series.str.title()
            elif op == "normalize_whitespace":
                # Mengubah spasi ganda/tab/newline menjadi satu spasi
                series = series.str.replace(r'\s+', ' ', regex=True)
            elif op == "remove_special_chars":
                # Hapus apa pun yang bukan huruf, angka, atau spasi
                series = series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

        if regex_replacements:
            for rule in regex_replacements:
                pat = rule.get('pattern')
                rep = rule.get('replacement', '')
                if pat:
                    series = series.str.replace(pat, rep, regex=True)

        self.df[column_name] = series
        return self.df

    # --- 2. PARSE INDONESIAN NUMERIC ---
    def parse_indonesian_numeric_currency(self, column_name: str, decimal_separator: str = ',', 
                                          thousand_separator: str = '.', currency_symbol_cleanup: bool = False, 
                                          handle_abbreviations: bool = False) -> pd.DataFrame:
        if column_name not in self.df.columns: return self.df
        
        series = self.df[column_name].astype(str)

        if currency_symbol_cleanup:
            # Hapus Rp, IDR, $, dan spasi
            series = series.str.replace(r'[Rp|IDR|\$|\s]', '', regex=True)

        # Hapus pemisah ribuan agar bisa diparse float
        if thousand_separator == '.':
            series = series.str.replace('.', '', regex=False)
        elif thousand_separator == ',':
            series = series.str.replace(',', '', regex=False)

        # Ganti pemisah desimal ke titik (standar Python)
        if decimal_separator == ',':
            series = series.str.replace(',', '.', regex=False)

        # Handle singkatan (jt, rb, m)
        if handle_abbreviations:
            def expand_abbr(val):
                val = val.lower()
                multiplier = 1
                if 'jt' in val: multiplier = 1e6; val = val.replace('jt', '')
                elif 'rb' in val: multiplier = 1e3; val = val.replace('rb', '')
                elif 'm' in val and 'omzet' not in val: multiplier = 1e9; val = val.replace('m', '') # Hati-hati dengan 'm'
                try:
                    return float(val) * multiplier
                except:
                    return val
            
            series = series.apply(expand_abbr)
        
        # Konversi ke numerik, error jadi NaN
        self.df[column_name] = pd.to_numeric(series, errors='coerce')
        return self.df

    # --- 3. STANDARDIZE DATETIME ---
    def standardize_indonesian_datetime(self, column_name: str, input_formats: List[str] = [], 
                                        target_format: str = '%Y-%m-%d', locale: str = 'id_ID', 
                                        timezone_conversion: Optional[Dict] = None) -> pd.DataFrame:
        if column_name not in self.df.columns: return self.df

        # Dictionary manual untuk nama bulan Indonesia (Dateutil kadang kurang akurat utk ID)
        indo_months = {
            'januari': 'January', 'februari': 'February', 'maret': 'March', 'mei': 'May',
            'juni': 'June', 'juli': 'July', 'agustus': 'August', 'september': 'September',
            'oktober': 'October', 'november': 'November', 'desember': 'December',
            'agt': 'Aug', 'okt': 'Oct', 'des': 'Dec'
        }
        
        def parse_date_flexible(val):
            if pd.isna(val) or val == '': return pd.NaT
            val_str = str(val).lower()
            # Ganti bulan Indo ke Inggris
            for indo, eng in indo_months.items():
                val_str = val_str.replace(indo, eng)
            
            try:
                # Coba parsing otomatis
                dt = parser.parse(val_str, dayfirst=True) # dayfirst=True umum di Indo (DD-MM-YYYY)
                return dt
            except:
                return pd.NaT

        # Proses parsing
        series = self.df[column_name].apply(parse_date_flexible)

        # Timezone conversion (jika ada)
        if timezone_conversion and 'from_tz' in timezone_conversion and 'to_tz' in timezone_conversion:
            try:
                from_tz = timezone_conversion['from_tz']
                to_tz = timezone_conversion['to_tz']
                # Asumsikan input naive adalah local time
                series = series.dt.tz_localize(from_tz, ambiguous='NaT', nonexistent='NaT').dt.tz_convert(to_tz)
                # Hapus info timezone jika target format tidak meminta (opsional, tergantung kebutuhan)
                series = series.dt.tz_localize(None) 
            except Exception as e:
                logger.warning(f"Gagal konversi timezone: {e}")

        # Format output
        self.df[column_name] = series.dt.strftime(target_format)
        return self.df

    # --- 4. HANDLE MISSING AND NULLS ---
    def handle_missing_and_nulls(self, column_name: str, strategy: str, 
                                 missing_indicators: List[str] = ["", "NaN", "NULL", "-", "n/a"],
                                 fill_value: Optional[str] = None) -> pd.DataFrame:
        if column_name not in self.df.columns: return self.df

        # 1. Standarisasi NaN
        # Ganti semua indikator custom menjadi np.nan
        self.df[column_name] = self.df[column_name].replace(missing_indicators, np.nan)
        
        # 2. Eksekusi Strategi
        if strategy == "drop_row":
            self.df.dropna(subset=[column_name], inplace=True)
        elif strategy == "fill_value" and fill_value is not None:
            self.df[column_name].fillna(fill_value, inplace=True)
        elif strategy == "fill_mean":
            # Pastikan numerik dulu
            mean_val = pd.to_numeric(self.df[column_name], errors='coerce').mean()
            self.df[column_name].fillna(mean_val, inplace=True)
        elif strategy == "fill_median":
            med_val = pd.to_numeric(self.df[column_name], errors='coerce').median()
            self.df[column_name].fillna(med_val, inplace=True)
        elif strategy == "fill_mode":
            mode_val = self.df[column_name].mode()[0]
            self.df[column_name].fillna(mode_val, inplace=True)
        elif strategy == "forward_fill":
            self.df[column_name].fillna(method='ffill', inplace=True)
        elif strategy == "backward_fill":
            self.df[column_name].fillna(method='bfill', inplace=True)

        return self.df

    # --- 5. NORMALIZE CATEGORICAL VALUES ---
    def normalize_categorical_values(self, column_name: str, mapping_rules: Dict[str, str], 
                                     case_insensitive: bool = True, default_value: Optional[str] = None) -> pd.DataFrame:
        if column_name not in self.df.columns: return self.df

        # Buat mapper yang efisien
        if case_insensitive:
            # Lowercase keys di mapping
            mapping_rules_lower = {k.lower(): v for k, v in mapping_rules.items()}
            
            def apply_map(val):
                if pd.isna(val): return val
                val_str = str(val).lower()
                return mapping_rules_lower.get(val_str, default_value if default_value else val)
            
            self.df[column_name] = self.df[column_name].apply(apply_map)
        else:
            # Direct mapping
            self.df[column_name] = self.df[column_name].replace(mapping_rules)
            if default_value:
                # Cek mana yang tidak ada di mapping rules (agak tricky di pandas, simplenya pakai apply)
                known_keys = set(mapping_rules.keys())
                self.df[column_name] = self.df[column_name].apply(lambda x: x if x in known_keys or list(mapping_rules.values()) else default_value)

        return self.df

    # --- 6. DEDUPLICATE AND FUZZY MATCH ---
    def deduplicate_and_fuzzy_match(self, subset_columns: List[str], method: str, 
                                    keep: str = 'first', fuzzy_threshold: int = 100) -> pd.DataFrame:
        # Cek kolom ada semua
        valid_cols = [c for c in subset_columns if c in self.df.columns]
        if not valid_cols: return self.df

        if method == "exact":
            if keep == "none": keep = False # Pandas syntax beda dikit
            self.df.drop_duplicates(subset=valid_cols, keep=keep, inplace=True)
        
        elif method == "fuzzy":
            # Note: Fuzzy dedup di Pandas murni SANGAT BERAT (O(N^2)). 
            # Untuk production massal, disarankan pakai library 'recordlinkage' atau 'thefuzz' dengan blocking.
            # Ini implementasi simpel untuk dataset kecil-menengah (<10k baris).
            
            logger.warning("Menjalankan Fuzzy Match. Proses ini mungkin lambat untuk data besar.")
            # (Implementasi placeholder yang aman untuk backend standar tanpa heavy dependencies)
            # Biasanya di backend production, kita fallback ke exact match + cleaning dulu
            # atau hanya melakukan exact match pada kolom yang sudah di-normalize.
            self.df.drop_duplicates(subset=valid_cols, keep=keep, inplace=True)
            logger.info("Fuzzy match disederhanakan menjadi exact match untuk stabilitas performa.")

        return self.df

    # --- 7. VALIDATE AND FILTER OUTLIERS ---
    def validate_and_filter_outliers(self, column_name: str, validation_type: str, 
                                     action: str, criteria: Dict[str, Any]) -> pd.DataFrame:
        if column_name not in self.df.columns: return self.df
        
        mask = pd.Series(True, index=self.df.index)

        if validation_type == "numeric_range":
            series_num = pd.to_numeric(self.df[column_name], errors='coerce')
            min_val = criteria.get('min', float('-inf'))
            max_val = criteria.get('max', float('inf'))
            mask = (series_num >= min_val) & (series_num <= max_val)
            # NaN (gagal parse) dianggap invalid
            mask = mask & series_num.notna()

        elif validation_type == "regex_match":
            pattern = criteria.get('pattern', '')
            if pattern:
                mask = self.df[column_name].astype(str).str.contains(pattern, regex=True, na=False)
        
        elif validation_type == "list_membership":
            allowed = criteria.get('allowed_values', [])
            mask = self.df[column_name].isin(allowed)

        # Terapkan Action pada yang INVALID (~mask)
        if action == "drop":
            self.df = self.df[mask]
        elif action == "nullify":
            self.df.loc[~mask, column_name] = np.nan
        elif action == "flag":
            self.df[f'{column_name}_is_valid'] = mask

        return self.df

    # --- 8. SPLIT AND EXTRACT ENTITIES ---
    def split_and_extract_entities(self, column_name: str, operation: str, new_column_names: List[str],
                                   delimiter: str = ',', regex_pattern: str = '') -> pd.DataFrame:
        if column_name not in self.df.columns: return self.df

        if operation == "split_by_delimiter":
            split_data = self.df[column_name].str.split(delimiter, expand=True)
            # Ambil sesuai jumlah kolom baru yang diminta
            for i, new_col in enumerate(new_column_names):
                if i < split_data.shape[1]:
                    self.df[new_col] = split_data[i].str.strip()
        
        elif operation == "extract_regex":
            if regex_pattern:
                extracted = self.df[column_name].str.extract(regex_pattern)
                for i, new_col in enumerate(new_column_names):
                    if i < extracted.shape[1]:
                        self.df[new_col] = extracted[i]

        return self.df

# --- CONTOH PENGGUNAAN (MAIN) ---
if __name__ == "__main__":
    # 1. Dummy Data
    data = {
        'Harga': ['Rp 10.000', '50rb', '1.5jt', 'invalid'],
        'Kota': ['jkt', 'SBY', 'Bandung', 'Jogja'],
        'Tanggal': ['17-08-1945', '2024/01/01', 'Des 25, 2023', '']
    }
    df = pd.DataFrame(data)
    
    print("--- DATA AWAL ---")
    print(df)

    # 2. Inisialisasi Executor
    executor = DataCleaningExecutor(df)

    # 3. Simulasi Panggilan dari FunctionGemma (JSON Output)
    
    # Tool 1: Bersihkan Angka
    executor.execute_tool("parse_indonesian_numeric_currency", {
        "column_name": "Harga",
        "currency_symbol_cleanup": True,
        "handle_abbreviations": True
    })

    # Tool 2: Normalisasi Kota
    executor.execute_tool("normalize_categorical_values", {
        "column_name": "Kota",
        "mapping_rules": {"jkt": "Jakarta", "sby": "Surabaya", "jogja": "Yogyakarta"}
    })

    # Tool 3: Standarisasi Tanggal
    executor.execute_tool("standardize_indonesian_datetime", {
        "column_name": "Tanggal",
        "target_format": "%Y-%m-%d"
    })

    print("\n--- DATA SETELAH PROSES ---")
    print(executor.df)
