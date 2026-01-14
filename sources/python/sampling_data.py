import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import random


def analyze_dataset_characteristics(parquet_dir):
    """
    Phân tích đặc điểm của dataset để xác định:
    - Hãng bay lớn
    - Sân bay lớn
    - Giờ cao điểm
    """
    print("🔍 Đang phân tích dataset để xác định đặc điểm...")
    
    all_files = list(Path(parquet_dir).rglob("*.parquet"))
    print(f"   Tìm thấy {len(all_files):,} file Parquet")
    
    # Lấy sample nhỏ để phân tích (đọc 1 file mỗi tháng)
    sample_files = []
    for month in range(1, 13):
        month_files = [f for f in all_files if f"month={month:02d}" in str(f)]
        if month_files:
            sample_files.append(month_files[0])
    
    print(f"   Đang đọc {len(sample_files)} file để phân tích...")
    
    dfs = []
    for file in sample_files[:3]:  # Chỉ đọc 3 file đầu để phân tích nhanh
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"   Lỗi đọc {file}: {e}")
    
    if not dfs:
        print("❌ Không đọc được file nào!")
        return None
    
    analysis_df = pd.concat(dfs, ignore_index=True)
    print(f"   Đã đọc {len(analysis_df):,} dòng để phân tích")
    
    # 1. Xác định hãng bay lớn (dựa trên số chuyến)
    print("\n📊 Phân tích hãng bay...")
    carrier_counts = analysis_df['OP_CARRIER'].value_counts()
    top_carriers = carrier_counts.head(10).index.tolist()
    print(f"   Top 10 hãng bay: {top_carriers}")
    
    # 2. Xác định sân bay lớn (dựa trên số chuyến xuất phát)
    print("\n🏢 Phân tích sân bay...")
    origin_counts = analysis_df['ORIGIN'].value_counts()
    dest_counts = analysis_df['DEST'].value_counts()
    
    # Kết hợp cả xuất phát và đến
    all_airports = pd.concat([origin_counts, dest_counts])
    airport_totals = all_airports.groupby(all_airports.index).sum()
    top_airports = airport_totals.sort_values(ascending=False).head(15).index.tolist()
    print(f"   Top 15 sân bay: {top_airports[:5]}... (tổng {len(top_airports)} sân bay)")
    
    # 3. Xác định giờ cao điểm
    print("\n⏰ Phân tích giờ cao điểm...")
    if 'DEP_TIME' in analysis_df.columns:
        # Chuyển DEP_TIME thành giờ
        analysis_df['DEP_HOUR'] = analysis_df['DEP_TIME'].astype(str).str[:2].fillna('00').astype(int)
        hour_counts = analysis_df['DEP_HOUR'].value_counts().sort_index()
        
        # Giờ cao điểm: 6-9h sáng và 16-19h tối
        peak_hours = list(range(6, 10)) + list(range(16, 20))
        print(f"   Giờ cao điểm xác định: {peak_hours}")
    else:
        peak_hours = [7, 8, 9, 17, 18, 19]  # Mặc định
        print(f"   Sử dụng giờ cao điểm mặc định: {peak_hours}")
    
    return {
        'top_carriers': top_carriers,
        'top_airports': top_airports,
        'peak_hours': peak_hours,
        'total_files': len(all_files)
    }


def stratified_sampling_from_parquet(parquet_dir, output_csv, target_rows=6000, 
                                     sample_per_month=500, random_seed=42):
    """
    Lấy mẫu phân tầng từ các file Parquet đã phân vùng
    
    Parameters:
    -----------
    parquet_dir : str
        Thư mục chứa các file Parquet (cấu trúc year=2023/month=01/...)
    output_csv : str
        Đường dẫn file CSV output
    target_rows : int
        Tổng số dòng mục tiêu (~6000)
    sample_per_month : int
        Số dòng lấy mỗi tháng (~500)
    random_seed : int
        Seed cho random để đảm bảo reproducibility
    """
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print("=" * 70)
    print("🎯 BẮT ĐẦU STRATIFIED SAMPLING")
    print("=" * 70)
    
    # 1. Phân tích dataset
    characteristics = analyze_dataset_characteristics(parquet_dir)
    if not characteristics:
        print("❌ Không thể phân tích dataset!")
        return
    
    top_carriers = characteristics['top_carriers']
    top_airports = characteristics['top_airports']
    peak_hours = characteristics['peak_hours']
    
    print(f"\n📈 Đặc điểm dataset:")
    print(f"   - Hãng bay lớn: {len(top_carriers)} hãng")
    print(f"   - Sân bay lớn: {len(top_airports)} sân bay")
    print(f"   - Giờ cao điểm: {peak_hours}")
    
    # 2. Tạo thư mục output nếu chưa tồn tại
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_sampled_data = []
    
    # 3. Xử lý từng tháng
    for month in range(1, 13):
        print(f"\n{'─' * 50}")
        print(f"📅 Xử lý tháng {month:02d}")
        print(f"{'─' * 50}")
        
        month_dir = Path(parquet_dir) / f"year=2023" / f"month={month:02d}"
        
        if not month_dir.exists():
            print(f"   ⚠️ Thư mục {month_dir} không tồn tại, bỏ qua...")
            continue
        
        # Đọc tất cả file Parquet trong tháng
        parquet_files = list(month_dir.glob("*.parquet"))
        print(f"   📁 Tìm thấy {len(parquet_files)} file Parquet")
        
        if not parquet_files:
            print(f"   ⚠️ Không có file Parquet trong tháng {month:02d}")
            continue
        
        # Đọc và kết hợp tất cả file trong tháng
        monthly_data = []
        for file_idx, parquet_file in enumerate(parquet_files):
            try:
                df = pd.read_parquet(parquet_file)
                monthly_data.append(df)
                
                if (file_idx + 1) % 5 == 0 or file_idx == len(parquet_files) - 1:
                    print(f"   📖 Đã đọc {file_idx + 1}/{len(parquet_files)} file...")
                    
            except Exception as e:
                print(f"   ❌ Lỗi đọc {parquet_file.name}: {e}")
                continue
        
        if not monthly_data:
            print(f"   ❌ Không đọc được file nào trong tháng {month:02d}")
            continue
        
        month_df = pd.concat(monthly_data, ignore_index=True)
        print(f"   📊 Tổng dòng trong tháng {month:02d}: {len(month_df):,}")
        
        # 4. Phân loại theo độ trễ
        print(f"   📊 Phân loại độ trễ...")
        
        # Chuẩn bị các trường cần thiết
        if 'DEP_TIME' in month_df.columns:
            month_df['DEP_HOUR'] = month_df['DEP_TIME'].astype(str).str[:2].fillna('00').astype(int)
        else:
            month_df['DEP_HOUR'] = 0
        
        if 'ARR_DELAY' not in month_df.columns:
            print(f"   ❌ Không có cột ARR_DELAY trong dữ liệu")
            continue
        
        # Loại bỏ các dòng không có ARR_DELAY
        month_df_clean = month_df.dropna(subset=['ARR_DELAY'])
        print(f"   🧹 Sau khi làm sạch: {len(month_df_clean):,} dòng")
        
        # Phân loại
        severe_delay = month_df_clean[month_df_clean['ARR_DELAY'] > 60]
        moderate_delay = month_df_clean[(month_df_clean['ARR_DELAY'] > 30) & (month_df_clean['ARR_DELAY'] <= 120)]
        minor_delay = month_df_clean[(month_df_clean['ARR_DELAY'] > 0) & (month_df_clean['ARR_DELAY'] <= 30)]
        on_time = month_df_clean[month_df_clean['ARR_DELAY'] <= 0]
        
        print(f"   📊 Phân bố độ trễ:")
        print(f"     - Trễ nhiều (>60p): {len(severe_delay):,} dòng")
        print(f"     - Trễ vừa (30-120p): {len(moderate_delay):,} dòng")
        print(f"     - Trễ ít (0-30p): {len(minor_delay):,} dòng")
        print(f"     - Đúng giờ (≤0p): {len(on_time):,} dòng")
        
        # 5. Lấy mẫu phân tầng cho tháng này
        print(f"\n   🎯 Lấy mẫu phân tầng...")
        
        sampled_month = []
        
        # 5.1. Lấy mẫu TRỄ NHIỀU (50 dòng)
        print(f"     🟥 Trễ nhiều: ", end="")
        if len(severe_delay) > 0:
            # Ưu tiên: hãng lớn, sân bay lớn, thời tiết xấu
            severe_delay['score'] = (
                severe_delay['OP_CARRIER'].apply(lambda x: 2 if x in top_carriers else 0) +
                severe_delay['ORIGIN'].apply(lambda x: 1.5 if x in top_airports else 0) +
                severe_delay['DEST'].apply(lambda x: 1.5 if x in top_airports else 0) +
                severe_delay.apply(lambda row: 2 if (pd.notna(row.get('O_PRCP')) and row.get('O_PRCP', 0) > 0) else 0, axis=1) +
                severe_delay.apply(lambda row: 2 if (pd.notna(row.get('O_WSPD')) and row.get('O_WSPD', 0) > 20) else 0, axis=1)
            )
            
            # Sắp xếp theo độ trẻ giảm dần và score
            severe_delay_sorted = severe_delay.sort_values(
                by=['ARR_DELAY', 'score'], 
                ascending=[False, False]
            )
            
            # Lấy 50 dòng, nhưng đảm bảo có đủ hãng
            n_severe = min(50, len(severe_delay_sorted))
            
            if n_severe > 0:
                # Đảm bảo mỗi hãng lớn có ít nhất 1-2 chuyến
                carriers_in_severe = severe_delay_sorted['OP_CARRIER'].unique()
                severe_samples = []
                
                for carrier in top_carriers[:5]:  # Top 5 hãng
                    carrier_flights = severe_delay_sorted[severe_delay_sorted['OP_CARRIER'] == carrier]
                    if len(carrier_flights) > 0:
                        n_carrier = min(3, len(carrier_flights))
                        severe_samples.append(carrier_flights.head(n_carrier))
                
                # Thêm các chuyến khác để đủ số lượng
                remaining = n_severe - sum(len(df) for df in severe_samples)
                if remaining > 0:
                    already_taken = pd.concat(severe_samples) if severe_samples else pd.DataFrame()
                    remaining_flights = severe_delay_sorted[~severe_delay_sorted.index.isin(already_taken.index)]
                    if len(remaining_flights) > 0:
                        additional = remaining_flights.head(remaining)
                        severe_samples.append(additional)
                
                if severe_samples:
                    severe_sampled = pd.concat(severe_samples, ignore_index=True)
                    sampled_month.append(severe_sampled)
                    print(f"Lấy {len(severe_sampled)} dòng")
                else:
                    print(f"Không lấy được")
            else:
                print(f"Không đủ dữ liệu")
        else:
            print(f"Không có dữ liệu")
        
        # 5.2. Lấy mẫu TRỄ VỪA (100 dòng)
        print(f"     🟧 Trễ vừa: ", end="")
        if len(moderate_delay) > 0:
            # Ưu tiên: giờ cao điểm, sân bay đông, thời tiết xấu
            moderate_delay['score'] = (
                moderate_delay['DEP_HOUR'].apply(lambda x: 2 if x in peak_hours else 0) +
                moderate_delay['ORIGIN'].apply(lambda x: 1.5 if x in top_airports else 0) +
                moderate_delay.apply(lambda row: 2 if (pd.notna(row.get('O_PRCP')) and row.get('O_PRCP', 0) > 0) else 0, axis=1)
            )
            
            moderate_delay_sorted = moderate_delay.sort_values(
                by=['score', 'ARR_DELAY'], 
                ascending=[False, False]
            )
            
            n_moderate = min(100, len(moderate_delay_sorted))
            if n_moderate > 0:
                # Đảm bảo phân bố theo hãng
                moderate_samples = []
                carriers = moderate_delay_sorted['OP_CARRIER'].unique()
                
                for carrier in carriers[:8]:  # Lấy 8 hãng đầu
                    carrier_flights = moderate_delay_sorted[moderate_delay_sorted['OP_CARRIER'] == carrier]
                    if len(carrier_flights) > 0:
                        n_carrier = min(15, len(carrier_flights))
                        moderate_samples.append(carrier_flights.head(n_carrier))
                
                moderate_sampled = pd.concat(moderate_samples, ignore_index=True)
                
                # Nếu chưa đủ, lấy thêm ngẫu nhiên
                if len(moderate_sampled) < n_moderate:
                    remaining = moderate_delay_sorted[~moderate_delay_sorted.index.isin(moderate_sampled.index)]
                    if len(remaining) > 0:
                        additional = remaining.head(n_moderate - len(moderate_sampled))
                        moderate_sampled = pd.concat([moderate_sampled, additional], ignore_index=True)
                
                moderate_sampled = moderate_sampled.head(n_moderate)
                sampled_month.append(moderate_sampled)
                print(f"Lấy {len(moderate_sampled)} dòng")
            else:
                print(f"Không đủ dữ liệu")
        else:
            print(f"Không có dữ liệu")
        
        # 5.3. Lấy mẫu TRỄ ÍT (150 dòng)
        print(f"     🟨 Trễ ít: ", end="")
        if len(minor_delay) > 0:
            # Đại diện các hãng, sân bay, giờ khác nhau
            n_minor = min(150, len(minor_delay))
            
            if n_minor > 0:
                # Phân tầng theo hãng
                minor_samples = []
                carriers = minor_delay['OP_CARRIER'].unique()
                
                for carrier in carriers:
                    carrier_flights = minor_delay[minor_delay['OP_CARRIER'] == carrier]
                    if len(carrier_flights) > 0:
                        # Tính tỷ lệ dựa trên số lượng chuyến của hãng
                        proportion = len(carrier_flights) / len(minor_delay)
                        n_from_carrier = max(2, int(n_minor * proportion))
                        n_from_carrier = min(n_from_carrier, len(carrier_flights))
                        
                        if n_from_carrier > 0:
                            sampled = carrier_flights.sample(n=n_from_carrier, random_state=random_seed)
                            minor_samples.append(sampled)
                
                if minor_samples:
                    minor_sampled = pd.concat(minor_samples, ignore_index=True)
                    
                    # Nếu chưa đủ, lấy thêm ngẫu nhiên
                    if len(minor_sampled) < n_minor:
                        remaining = n_minor - len(minor_sampled)
                        all_minor = minor_delay[~minor_delay.index.isin(minor_sampled.index)]
                        if len(all_minor) > 0:
                            additional = all_minor.sample(n=min(remaining, len(all_minor)), random_state=random_seed)
                            minor_sampled = pd.concat([minor_sampled, additional], ignore_index=True)
                    
                    minor_sampled = minor_sampled.head(n_minor)
                    sampled_month.append(minor_sampled)
                    print(f"Lấy {len(minor_sampled)} dòng")
                else:
                    print(f"Không lấy được")
            else:
                print(f"Không đủ dữ liệu")
        else:
            print(f"Không có dữ liệu")
        
        # 5.4. Lấy mẫu ĐÚNG GIỜ (200 dòng)
        print(f"     🟩 Đúng giờ: ", end="")
        if len(on_time) > 0:
            # Ưu tiên: giờ cao điểm, thời tiết xấu mà vẫn đúng giờ
            on_time['score'] = (
                on_time['DEP_HOUR'].apply(lambda x: 3 if x in peak_hours else 0) +
                on_time.apply(lambda row: 3 if (pd.notna(row.get('O_PRCP')) and row.get('O_PRCP', 0) > 0) else 0, axis=1) +
                on_time.apply(lambda row: 3 if (pd.notna(row.get('O_WSPD')) and row.get('O_WSPD', 0) > 20) else 0, axis=1)
            )
            
            on_time_sorted = on_time.sort_values(by=['score'], ascending=False)
            
            n_ontime = min(200, len(on_time_sorted))
            
            if n_ontime > 0:
                # Lấy các chuyến đặc biệt trước
                special_cases = on_time_sorted[on_time_sorted['score'] > 0]
                n_special = min(50, len(special_cases))
                
                if n_special > 0:
                    special_sampled = special_cases.head(n_special)
                else:
                    special_sampled = pd.DataFrame()
                
                # Lấy thêm các chuyến thông thường
                remaining_needed = n_ontime - len(special_sampled)
                if remaining_needed > 0:
                    regular_cases = on_time_sorted[on_time_sorted['score'] == 0]
                    if len(regular_cases) > 0:
                        regular_sampled = regular_cases.sample(
                            n=min(remaining_needed, len(regular_cases)), 
                            random_state=random_seed
                        )
                    else:
                        regular_sampled = pd.DataFrame()
                
                if not special_sampled.empty or not regular_sampled.empty:
                    ontime_sampled = pd.concat([special_sampled, regular_sampled], ignore_index=True)
                    ontime_sampled = ontime_sampled.head(n_ontime)
                    sampled_month.append(ontime_sampled)
                    print(f"Lấy {len(ontime_sampled)} dòng")
                else:
                    print(f"Không lấy được")
            else:
                print(f"Không đủ dữ liệu")
        else:
            print(f"Không có dữ liệu")
        
        # 6. Kết hợp tất cả mẫu của tháng
        if sampled_month:
            month_sampled = pd.concat(sampled_month, ignore_index=True)
            
            # Giới hạn số dòng mỗi tháng (~500)
            month_sampled = month_sampled.head(sample_per_month)
            
            # Thêm cột tháng để theo dõi
            month_sampled['SAMPLING_MONTH'] = month
            
            all_sampled_data.append(month_sampled)
            
            print(f"\n   ✅ Tháng {month:02d}: Lấy được {len(month_sampled)} dòng")
            print(f"      Phân bố: Trễ nhiều: {len(month_sampled[month_sampled['ARR_DELAY'] > 60])}, "
                  f"Trễ vừa: {len(month_sampled[(month_sampled['ARR_DELAY'] > 30) & (month_sampled['ARR_DELAY'] <= 120)])}, "
                  f"Trễ ít: {len(month_sampled[(month_sampled['ARR_DELAY'] > 0) & (month_sampled['ARR_DELAY'] <= 30)])}, "
                  f"Đúng giờ: {len(month_sampled[month_sampled['ARR_DELAY'] <= 0])}")
        else:
            print(f"\n   ❌ Tháng {month:02d}: Không lấy được dữ liệu nào")
    
    # 7. Kết hợp tất cả tháng
    print(f"\n{'=' * 70}")
    print("📦 KẾT HỢP KẾT QUẢ")
    print(f"{'=' * 70}")
    
    if not all_sampled_data:
        print("❌ Không có dữ liệu nào được lấy mẫu!")
        return
    
    final_sampled = pd.concat(all_sampled_data, ignore_index=True)
    
    # Xóa cột score nếu có
    if 'score' in final_sampled.columns:
        final_sampled = final_sampled.drop(columns=['score'])
    
    # Xóa cột DEP_HOUR tạm thời nếu có
    if 'DEP_HOUR' in final_sampled.columns:
        final_sampled = final_sampled.drop(columns=['DEP_HOUR'])
    
    print(f"📊 Tổng số dòng sampling: {len(final_sampled):,}")
    
    # 8. Phân tích tính cân bằng
    print(f"\n📈 PHÂN TÍCH TÍNH CÂN BẰNG:")
    
    # 8.1. Phân bố theo tháng
    print(f"\n   📅 Phân bố theo tháng:")
    month_dist = final_sampled['SAMPLING_MONTH'].value_counts().sort_index()
    for month, count in month_dist.items():
        print(f"      - Tháng {month:02d}: {count:3d} dòng")
    
    # 8.2. Phân bố theo hãng bay
    print(f"\n   ✈️ Phân bố theo hãng bay:")
    carrier_dist = final_sampled['OP_CARRIER'].value_counts()
    print(f"      Tổng số hãng: {len(carrier_dist)}")
    for carrier, count in carrier_dist.head(10).items():
        print(f"      - {carrier}: {count:3d} chuyến")
    
    # 8.3. Phân bố theo sân bay
    print(f"\n   🏢 Phân bố theo sân bay (xuất phát):")
    origin_dist = final_sampled['ORIGIN'].value_counts()
    print(f"      Tổng số sân bay: {len(origin_dist)}")
    
    # Kiểm tra có bao nhiêu sân bay lớn được bao phủ
    covered_top_airports = [ap for ap in top_airports if ap in origin_dist.index]
    print(f"      Sân bay lớn được bao phủ: {len(covered_top_airports)}/{len(top_airports)}")
    
    # 8.4. Phân bố theo độ trễ
    print(f"\n   ⏱️ Phân bố theo độ trễ:")
    delay_categories = {
        'Trễ nhiều (>60p)': len(final_sampled[final_sampled['ARR_DELAY'] > 60]),
        'Trễ vừa (30-120p)': len(final_sampled[(final_sampled['ARR_DELAY'] > 30) & (final_sampled['ARR_DELAY'] <= 120)]),
        'Trễ ít (0-30p)': len(final_sampled[(final_sampled['ARR_DELAY'] > 0) & (final_sampled['ARR_DELAY'] <= 30)]),
        'Đúng giờ (≤0p)': len(final_sampled[final_sampled['ARR_DELAY'] <= 0])
    }
    
    for category, count in delay_categories.items():
        percentage = (count / len(final_sampled)) * 100
        print(f"      - {category}: {count:3d} dòng ({percentage:.1f}%)")
    
    # 8.5. Kiểm tra các trường hợp có thể so sánh
    print(f"\n   🔄 Kiểm tra khả năng so sánh:")
    
    # Đếm số cặp cùng tuyến bay
    route_counts = final_sampled.groupby(['ORIGIN', 'DEST']).size()
    routes_with_multiple = (route_counts > 1).sum()
    print(f"      - Số tuyến bay có >1 chuyến: {routes_with_multiple}")
    
    # Đếm số cặp cùng hãng cùng giờ
    if 'DEP_TIME' in final_sampled.columns:
        final_sampled['DEP_HOUR'] = final_sampled['DEP_TIME'].astype(str).str[:2].fillna('00').astype(int)
        hour_carrier_counts = final_sampled.groupby(['OP_CARRIER', 'DEP_HOUR']).size()
        hour_carrier_pairs = (hour_carrier_counts > 1).sum()
        print(f"      - Số cặp hãng-giờ có >1 chuyến: {hour_carrier_pairs}")
    
    # 9. Lưu file CSV
    print(f"\n💾 Đang lưu file CSV...")
    final_sampled.to_csv(output_csv, index=False, encoding='utf-8')
    
    file_size_mb = Path(output_csv).stat().st_size / (1024 ** 2)
    print(f"✅ Đã lưu: {output_csv}")
    print(f"📏 Kích thước file: {file_size_mb:.2f} MB")
    print(f"👥 Tổng số dòng: {len(final_sampled):,}")
    
    # 10. Lưu report
    report_path = output_path.parent / "sampling_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SAMPLING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Sampling date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {parquet_dir}\n")
        f.write(f"Output file: {output_csv}\n")
        f.write(f"Total rows sampled: {len(final_sampled)}\n\n")
        
        f.write("MONTHLY DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for month, count in month_dist.items():
            f.write(f"Month {month:02d}: {count:3d} rows\n")
        
        f.write("\nCARRIER DISTRIBUTION (Top 15):\n")
        f.write("-" * 30 + "\n")
        for carrier, count in carrier_dist.head(15).items():
            f.write(f"{carrier}: {count:3d} flights\n")
        
        f.write("\nDELAY CATEGORY DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for category, count in delay_categories.items():
            percentage = (count / len(final_sampled)) * 100
            f.write(f"{category}: {count:3d} rows ({percentage:.1f}%)\n")
        
        f.write("\nBALANCE CHECK:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Top carriers covered: {len(set(top_carriers) & set(carrier_dist.index))}/{len(top_carriers)}\n")
        f.write(f"Top airports covered: {len(covered_top_airports)}/{len(top_airports)}\n")
        f.write(f"Routes with comparisons: {routes_with_multiple}\n")
        
        if 'hour_carrier_pairs' in locals():
            f.write(f"Hour-carrier pairs: {hour_carrier_pairs}\n")
    
    print(f"📝 Đã lưu báo cáo: {report_path}")
    print(f"\n🎉 HOÀN THÀNH STRATIFIED SAMPLING!")


# Main execution
if __name__ == "__main__":
    # Cấu hình
    PARQUET_DIR = "D:/UEL/DA_AVD/ADA-Flight-Delay/data/parquet/flights_weather"
    OUTPUT_CSV = "D:/UEL/DA_AVD/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    
    # Chạy sampling
    stratified_sampling_from_parquet(
        parquet_dir=PARQUET_DIR,
        output_csv=OUTPUT_CSV,
        target_rows=6000,
        sample_per_month=500,
        random_seed=42
    )