import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
import warnings

# Tắt cảnh báo không cần thiết
warnings.simplefilter(action='ignore', category=FutureWarning)

def validate_sampling_quality(parquet_dir, sample_csv_path):
    """
    Kiểm định tính đại diện (Phiên bản Robust - Chống lỗi Schema Mismatch).
    """
    
    # ---------------------------------------------------------
    # 0. THIẾT LẬP
    # ---------------------------------------------------------
    sample_path = Path(sample_csv_path)
    report_dir = sample_path.parent.parent / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_txt_path = report_dir / 'validation_report.txt'
    
    log_buffer = []

    def log(text=""):
        print(text)
        log_buffer.append(str(text))

    log(f"🚀 BẮT ĐẦU KIỂM ĐỊNH (VALIDATION)...")
    log(f"📂 Report Directory: {report_dir}")
    log("-" * 60)

    # ---------------------------------------------------------
    # BƯỚC 1: LOAD DỮ LIỆU
    # ---------------------------------------------------------
    
    # 1.1 Đọc Sample
    log("1️⃣ Đang đọc dữ liệu Mẫu (Sample)...")
    try:
        df_sample = pd.read_csv(sample_csv_path)
        log(f"   -> Đã load {len(df_sample):,} dòng mẫu.")
    except Exception as e:
        log(f"❌ Lỗi đọc file mẫu: {e}")
        return

    # 1.2 Đọc Population (FIX LỖI SCHEMA Ở ĐÂY)
    log("2️⃣ Đang đọc dữ liệu Tổng thể (Population)...")
    log("   -> Chuyển sang chế độ đọc từng file (Iterative Read) để tránh lỗi Schema...")
    
    try:
        # Tìm tất cả file parquet
        files = list(Path(parquet_dir).rglob("*.parquet"))
        if not files:
            log("❌ Không tìm thấy file parquet nào!")
            return
        
        log(f"   -> Tìm thấy {len(files)} file Parquet. Đang xử lý...")
        
        cols_needed = ['OP_CARRIER', 'FL_DATE', 'ORIGIN', 'DEP_TIME']
        dfs = []
        
        # Loop qua từng file để đọc
        for i, f in enumerate(files):
            try:
                # Chỉ đọc các cột cần thiết, bỏ qua cột 'month' gây lỗi
                df_part = pd.read_parquet(f, columns=cols_needed)
                dfs.append(df_part)
                
                # In tiến độ mỗi 20 file
                if (i + 1) % 20 == 0:
                    print(f"      ... Đã đọc {i + 1}/{len(files)} file")
            except Exception as e:
                # Nếu file nào lỗi quá thì bỏ qua, in warning nhẹ
                continue
        
        if not dfs:
            log("❌ Không đọc được file nào thành công.")
            return

        # Gộp lại thành 1 DataFrame lớn
        df_pop = pd.concat(dfs, ignore_index=True)
        
        # Xử lý ngày tháng (Tự tính lại Month/Day để đảm bảo nhất quán)
        if 'FL_DATE' in df_pop.columns:
            df_pop['FL_DATE'] = pd.to_datetime(df_pop['FL_DATE'])
            df_pop['MONTH'] = df_pop['FL_DATE'].dt.month
            df_pop['DAY_OF_WEEK'] = df_pop['FL_DATE'].dt.dayofweek + 1
        
        # Xử lý giờ bay
        def extract_hour(df):
            if 'DEP_TIME' in df.columns:
                return pd.to_numeric(df['DEP_TIME'], errors='coerce').fillna(0).astype(int) // 100
            return None

        df_pop['DEP_HOUR'] = extract_hour(df_pop)
        df_sample['DEP_HOUR'] = extract_hour(df_sample)
            
        log(f"   -> ✅ Đã load thành công {len(df_pop):,} dòng tổng thể.")
        
    except Exception as e:
        log(f"❌ Lỗi nghiêm trọng khi đọc Population: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # BƯỚC 2: VẼ VÀ LƯU BIỂU ĐỒ
    # ---------------------------------------------------------
    log("\n🎨 Đang vẽ và lưu biểu đồ...")

    try:
        # --- Chart 1: Carrier ---
        pop_carrier = df_pop['OP_CARRIER'].value_counts(normalize=True).reset_index()
        pop_carrier.columns = ['Carrier', 'Percentage']
        pop_carrier['Type'] = 'Population'
        
        sample_carrier = df_sample['OP_CARRIER'].value_counts(normalize=True).reset_index()
        sample_carrier.columns = ['Carrier', 'Percentage']
        sample_carrier['Type'] = 'Sample'
        
        comp_carrier = pd.concat([pop_carrier, sample_carrier])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=comp_carrier, x='Carrier', y='Percentage', hue='Type', palette=['#bdc3c7', '#e74c3c'])
        plt.title('Validation 1: Carrier Distribution', fontsize=14, fontweight='bold')
        plt.savefig(report_dir / 'val_1_carrier.png', bbox_inches='tight')
        plt.close()

        # --- Chart 2: Airport ---
        top_20_airports = df_pop['ORIGIN'].value_counts().head(20).index.tolist()
        pop_airport = df_pop[df_pop['ORIGIN'].isin(top_20_airports)]['ORIGIN'].value_counts(normalize=True).reset_index()
        pop_airport.columns = ['Airport', 'Percentage']
        pop_airport['Type'] = 'Population'
        sample_airport = df_sample[df_sample['ORIGIN'].isin(top_20_airports)]['ORIGIN'].value_counts(normalize=True).reset_index()
        sample_airport.columns = ['Airport', 'Percentage']
        sample_airport['Type'] = 'Sample'
        comp_airport = pd.concat([pop_airport, sample_airport])
        
        plt.figure(figsize=(14, 6))
        sns.barplot(data=comp_airport, x='Airport', y='Percentage', hue='Type', palette=['#bdc3c7', '#3498db'])
        plt.title('Validation 2: Top 20 Origin Airports Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.savefig(report_dir / 'val_2_airport.png', bbox_inches='tight')
        plt.close()

        # --- Chart 3: Peak Hour ---
        pop_hour = df_pop['DEP_HOUR'].value_counts(normalize=True).sort_index().reset_index()
        pop_hour.columns = ['Hour', 'Percentage']
        pop_hour['Type'] = 'Population'
        sample_hour = df_sample['DEP_HOUR'].value_counts(normalize=True).sort_index().reset_index()
        sample_hour.columns = ['Hour', 'Percentage']
        sample_hour['Type'] = 'Sample'
        comp_hour = pd.concat([pop_hour, sample_hour])
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=comp_hour, x='Hour', y='Percentage', hue='Type', style='Type', markers=True, dashes=False, palette=['gray', 'red'])
        plt.axvspan(6, 9, color='yellow', alpha=0.2, label='Peak Morning')
        plt.axvspan(16, 19, color='orange', alpha=0.2, label='Peak Afternoon')
        plt.title('Validation 3: Departure Hour Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(report_dir / 'val_3_peak_hour.png', bbox_inches='tight')
        plt.close()

        log("   ✅ Đã lưu 3 biểu đồ thành công.")

    except Exception as e:
        log(f"❌ Lỗi khi vẽ biểu đồ: {e}")

    # ---------------------------------------------------------
    # BƯỚC 3: GHI REPORT
    # ---------------------------------------------------------
    log("\n📋 KẾT QUẢ KIỂM ĐỊNH CHI TIẾT")
    log("=" * 60)
    
    # Temporal
    missing_months = set(range(1, 13)) - set(df_sample['MONTH'].unique())
    missing_days = set(range(1, 8)) - set(df_sample['DAY_OF_WEEK'].unique())
    log(f"1. TEMPORAL COVERAGE:")
    log(f"   - Months: {'✅ Đủ 12 tháng' if not missing_months else f'❌ Thiếu: {missing_months}'}")
    log(f"   - Days:   {'✅ Đủ 7 ngày' if not missing_days else f'❌ Thiếu: {missing_days}'}")

    # Peak Hour
    peak_hours = list(range(6, 10)) + list(range(16, 20))
    pop_peak_ratio = df_pop['DEP_HOUR'].isin(peak_hours).mean()
    sample_peak_ratio = df_sample['DEP_HOUR'].isin(peak_hours).mean()
    diff_peak = abs(pop_peak_ratio - sample_peak_ratio)
    log(f"\n2. PEAK HOUR (6-9h, 16-19h):")
    log(f"   - Pop: {pop_peak_ratio:.2%} | Sample: {sample_peak_ratio:.2%} | Diff: {diff_peak:.2%}")

    # Carriers
    log(f"\n3. CARRIER CONSISTENCY (Top 5):")
    merged = pd.merge(pop_carrier, sample_carrier, on='Carrier', suffixes=('_Pop', '_Sample'))
    merged['Diff'] = abs(merged['Percentage_Pop'] - merged['Percentage_Sample'])
    
    log(f"   {'Carrier':<8} | {'Pop %':<10} | {'Sample %':<10} | {'Diff %':<10}")
    log(f"   {'-'*45}")
    for _, row in merged.head(5).iterrows():
        log(f"   {row['Carrier']:<8} | {row['Percentage_Pop']:.2%}     | {row['Percentage_Sample']:.2%}     | {row['Diff']:.2%}")

    # Save TXT
    try:
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_buffer))
        log(f"\n💾 Đã lưu báo cáo tại: {report_txt_path}")
    except Exception as e:
        print(f"❌ Lỗi lưu file txt: {e}")

# --- CHẠY CODE ---
if __name__ == "__main__":
    # ĐƯỜNG DẪN CỦA BẠN
    PARQUET_DIR = "D:/UEL/ADA/ADA-Flight-Delay/data/parquet/flights_weather/year=2023"
    SAMPLE_CSV = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    
    validate_sampling_quality(PARQUET_DIR, SAMPLE_CSV)