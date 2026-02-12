import pandas as pd
from pathlib import Path
import sys

def analyze_parquet_uniques(parquet_dir, columns_to_check):
    """
    Đọc từng file Parquet và tổng hợp các giá trị duy nhất (Unique Values).
    Giúp debug lỗi schema và đối chiếu dữ liệu.
    """
    print(f"🔍 BẮT ĐẦU PHÂN TÍCH: {parquet_dir}")
    print(f"🎯 Các cột cần kiểm tra: {columns_to_check}\n")

    # Tìm tất cả file parquet
    files = list(Path(parquet_dir).rglob("*.parquet"))
    if not files:
        print("❌ Không tìm thấy file Parquet nào!")
        return

    # Dictionary để lưu thống kê: { 'tên_cột': { 'giá_trị': số_lần_xuất_hiện } }
    master_counts = {col: {} for col in columns_to_check}
    
    # Dictionary để lưu kiểu dữ liệu: { 'tên_cột': set(các_kiểu_dữ_liệu_đã_gặp) }
    type_tracker = {col: set() for col in columns_to_check}

    total_rows = 0
    files_read = 0

    print(f"📂 Tìm thấy {len(files)} file. Đang xử lý...")

    for idx, file_path in enumerate(files):
        try:
            # Chỉ đọc các cột cần thiết để nhanh
            # Lưu ý: Nếu cột không tồn tại trong file, pandas sẽ báo lỗi, ta cần try-except
            try:
                df = pd.read_parquet(file_path, columns=columns_to_check)
            except Exception:
                # Fallback: Nếu file thiếu cột nào đó, đọc hết rồi lọc sau (chậm hơn chút)
                df = pd.read_parquet(file_path)
                missing_cols = [c for c in columns_to_check if c not in df.columns]
                if missing_cols:
                    # Bỏ qua file này hoặc warning nếu cần
                    continue
                df = df[columns_to_check]

            rows_in_file = len(df)
            total_rows += rows_in_file
            files_read += 1

            for col in columns_to_check:
                # 1. Check kiểu dữ liệu (để debug lỗi int vs dictionary)
                dtype = str(df[col].dtype)
                type_tracker[col].add(dtype)

                # 2. Đếm value counts trong file này
                v_counts = df[col].value_counts().to_dict()
                
                # 3. Cộng dồn vào master_counts
                for val, count in v_counts.items():
                    # Chuyển val về string để tránh lỗi hash key khác kiểu
                    val_key = str(val) 
                    if val_key in master_counts[col]:
                        master_counts[col][val_key] += count
                    else:
                        master_counts[col][val_key] = count

            # In tiến độ
            if (idx + 1) % 10 == 0:
                print(f"   ... Đã đọc {idx + 1}/{len(files)} file ({total_rows:,} dòng)")

        except Exception as e:
            print(f"⚠️ Lỗi đọc file {file_path.name}: {e}")

    print("\n" + "="*60)
    print(f"✅ HOÀN THÀNH! Tổng số dòng đã quét: {total_rows:,}")
    print("="*60)

    # --- HIỂN THỊ KẾT QUẢ ---
    for col in columns_to_check:
        print(f"\n📊 PHÂN TÍCH CỘT: [{col}]")
        print(f"   - Các kiểu dữ liệu đã gặp: {type_tracker[col]}")
        
        counts = master_counts[col]
        unique_count = len(counts)
        print(f"   - Số lượng giá trị duy nhất (Cardinality): {unique_count}")
        
        # Sắp xếp theo số lượng giảm dần
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        
        print(f"   - Top 10 giá trị xuất hiện nhiều nhất:")
        print(f"     {'Giá trị':<20} | {'Số lượng':<15} | {'Tỷ lệ %'}")
        print(f"     {'-'*20} | {'-'*15} | {'-'*10}")
        
        for val, count in sorted_counts[:15]: # Show top 15
            percent = (count / total_rows) * 100 if total_rows > 0 else 0
            print(f"     {str(val):<20} | {count:<15,} | {percent:.2f}%")
            
        if unique_count > 15:
            print(f"     ... và {unique_count - 15} giá trị khác.")

# --- CHẠY CODE ---
if __name__ == "__main__":
    # Đổi đường dẫn tới thư mục Parquet gốc của bạn
    PARQUET_DIR = "D:/UEL/ADA/ADA-Flight-Delay/data/parquet/flights_weather/year=2023"
    
    # Danh sách cột bạn muốn kiểm tra đối chiếu
    # Thêm 'month' vào đây để xem lỗi gì
    COLS_TO_CHECK = ['OP_CARRIER', 'ORIGIN', 'MONTH'] 
    
    # Nếu file gốc không có cột 'MONTH' (do partition), bạn nên check cột khác hoặc 'FL_DATE'
    # COLS_TO_CHECK = ['OP_CARRIER', 'ORIGIN'] 
    
    analyze_parquet_uniques(PARQUET_DIR, COLS_TO_CHECK)