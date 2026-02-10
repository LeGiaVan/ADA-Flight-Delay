import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class FlightSamplingPipeline:
    def __init__(self, input_parquet_dir, output_dir, year=2023):
        self.input_dir = Path(input_parquet_dir)
        self.output_dir = Path(output_dir)
        self.year = year
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- CẤU HÌNH THEO KHUNG LÝ THUYẾT ---
        
        # 1. Định mức mẫu (Quota Allocation) - Ref: Section 2.2
        self.QUOTAS = {
            'Severe':   {'min': 60,  'max': 99999, 'count': 50},  # > 60 mins
            'Moderate': {'min': 30,  'max': 60,    'count': 100}, # 30 - 60 mins
            'Minor':    {'min': 0,   'max': 30,    'count': 150}, # 0 - 30 mins
            'OnTime':   {'min': -999,'max': 0,     'count': 200}  # <= 0 mins
        }
        
        # 2. Trọng số ưu tiên (Weighting Weights) - Ref: Section 3.2
        self.WEIGHTS = {
            'carrier': 2.0,  # Ưu tiên hãng lớn
            'airport': 1.5,  # Ưu tiên sân bay lớn
            'peak': 1.0,     # Ưu tiên giờ cao điểm
            'base': 1.0      # Điểm cơ bản
        }
        
        # Giờ cao điểm (Peak Hours): 6-9h và 16-19h
        self.PEAK_HOURS = list(range(6, 10)) + list(range(16, 20))
        
        # Các biến lưu trữ Top Carrier/Airport (sẽ được fill khi chạy analyze)
        self.top_carriers = []
        self.top_airports = []

    def _analyze_population_characteristics(self):
        """
        Bước 1: Quét nhanh dữ liệu để xác định Top Carriers và Top Airports.
        Dùng để tính trọng số chính xác.
        """
        print(f"🔍 [Phase 1] Analyzing Population (Năm {self.year})...")
        
        # Đọc mẫu nhanh (chỉ cần lấy 1-2 tháng đại diện hoặc đọc metadata nếu có)
        # Ở đây ta đọc tháng 1 và tháng 7 để đại diện mùa thấp điểm/cao điểm
        sample_months = [1, 7]
        dfs = []
        
        for m in sample_months:
            p = self.input_dir / f"year={self.year}" / f"month={m:02d}"
            if p.exists():
                try:
                    # Chỉ đọc cột cần thiết để nhanh
                    df = pd.read_parquet(p, columns=['OP_CARRIER', 'ORIGIN', 'DEST'])
                    dfs.append(df)
                except Exception as e:
                    print(f"  ⚠️ Warning: Không đọc được tháng {m}: {e}")
        
        if not dfs:
            print("  ❌ Không tìm thấy dữ liệu mẫu. Dùng danh sách mặc định.")
            return

        full_sample = pd.concat(dfs)
        
        # Lấy Top 10 Hãng bay
        self.top_carriers = full_sample['OP_CARRIER'].value_counts().head(10).index.tolist()
        
        # Lấy Top 20 Sân bay (Origin + Dest)
        all_airports = pd.concat([full_sample['ORIGIN'], full_sample['DEST']])
        self.top_airports = all_airports.value_counts().head(20).index.tolist()
        
        print(f"  ✅ Identified {len(self.top_carriers)} Top Carriers & {len(self.top_airports)} Top Airports.")

    def _calculate_priority_score(self, df):
        """
        Bước 2: Tính điểm ưu tiên (Priority Score) cho từng dòng.
        Công thức: S = Base + w_carrier + w_airport + w_peak
        Ref: Section 3.2 Formula
        """
        # Khởi tạo điểm cơ bản
        scores = np.ones(len(df)) * self.WEIGHTS['base']
        
        # Cộng điểm Hãng bay (Vectorized operation)
        if 'OP_CARRIER' in df.columns:
            scores += np.where(df['OP_CARRIER'].isin(self.top_carriers), self.WEIGHTS['carrier'], 0)
            
        # Cộng điểm Sân bay (Origin hoặc Dest nằm trong top đều được cộng)
        if 'ORIGIN' in df.columns and 'DEST' in df.columns:
            is_top_origin = df['ORIGIN'].isin(self.top_airports)
            is_top_dest = df['DEST'].isin(self.top_airports)
            # Dùng logic OR: Chỉ cần 1 trong 2 là sân bay lớn thì được cộng điểm
            scores += np.where(is_top_origin | is_top_dest, self.WEIGHTS['airport'], 0)
            
        # Cộng điểm Giờ cao điểm
        if 'DEP_TIME' in df.columns:
            # Chuyển DEP_TIME (float/str) về giờ (int). VD: 630.0 -> 6
            dep_hour = pd.to_numeric(df['DEP_TIME'], errors='coerce').fillna(0).astype(int) // 100
            scores += np.where(np.isin(dep_hour, self.PEAK_HOURS), self.WEIGHTS['peak'], 0)
            
        return scores

    def _weighted_sampling(self, df, n_samples, random_state):
        """
        Hàm core: Thực hiện Weighted Random Sampling.
        Nếu dữ liệu ít hơn n_samples -> Lấy hết (không sinh thêm để tránh fake data).
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        if len(df) <= n_samples:
            return df  # Lấy hết nếu không đủ (Oversampling tự nhiên bằng cách giữ nguyên)
            
        # Normalization trọng số để hàm sample hiểu
        weights = df['priority_score']
        if weights.sum() == 0:
            weights = None # Fallback về random thường nếu weights lỗi
            
        return df.sample(n=n_samples, weights=weights, random_state=random_state)

    def run_pipeline(self):
        print(f"\n🚀 [Phase 2] Starting Stratified Sampling Pipeline...")
        
        # 1. Chạy phân tích trước
        self._analyze_population_characteristics()
        
        all_samples = []
        stats = []

        # 2. Loop qua 12 tháng (Stratified by Time)
        for month in range(1, 13):
            month_path = self.input_dir / f"year={self.year}" / f"month={month:02d}"
            
            if not month_path.exists():
                print(f"  ⚠️ Tháng {month:02d}: Không tìm thấy dữ liệu.")
                continue
                
            print(f"  📅 Processing Month {month:02d}...", end=" ")
            
            try:
                # Đọc dữ liệu tháng
                df = pd.read_parquet(month_path)
                
                # Preprocessing cơ bản
                if 'ARR_DELAY' not in df.columns:
                    print("Skipping (Missing ARR_DELAY)")
                    continue
                
                df = df.dropna(subset=['ARR_DELAY'])
                
                # Tính Score
                df['priority_score'] = self._calculate_priority_score(df)
                
                month_collected = []
                
                # 3. Loop qua từng tầng độ trễ (Stratified by Severity)
                for label, criteria in self.QUOTAS.items():
                    # Filter dữ liệu theo định nghĩa tầng
                    mask = (df['ARR_DELAY'] > criteria['min']) & (df['ARR_DELAY'] <= criteria['max'])
                    subset = df[mask]
                    
                    # 4. Lấy mẫu có trọng số (Weighted Sampling)
                    sampled = self._weighted_sampling(
                        subset, 
                        n_samples=criteria['count'], 
                        random_state=42 + month # Seed thay đổi theo tháng
                    )
                    
                    # Gán nhãn để theo dõi sau này
                    sampled['DELAY_GROUP'] = label
                    sampled['SAMPLING_MONTH'] = month
                    
                    month_collected.append(sampled)
                    
                    # Log thống kê
                    stats.append({
                        'Month': month,
                        'Group': label,
                        'Available': len(subset),
                        'Sampled': len(sampled)
                    })

                # Gộp mẫu tháng
                month_final = pd.concat(month_collected)
                all_samples.append(month_final)
                print(f"✅ OK (Selected {len(month_final)} rows)")
                
            except Exception as e:
                print(f"❌ Error: {e}")

        # 5. Kết hợp và Xuất dữ liệu
        print(f"\n📦 [Phase 3] Exporting Data...")
        if not all_samples:
            print("❌ No data sampled!")
            return

        final_df = pd.concat(all_samples, ignore_index=True)
        
        # Clean up cột tạm
        cols_to_drop = ['priority_score']
        final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], inplace=True)
        
        # Save CSV
        output_file = self.output_dir / f"flight_data_sampled_{self.year}.csv"
        final_df.to_csv(output_file, index=False)
        
        # Save Report
        self._save_report(final_df, stats)
        
        print(f"🎉 Pipeline Completed Successfully!")
        print(f"   - Output: {output_file}")
        print(f"   - Total Rows: {len(final_df)}")

    def _save_report(self, df, stats):
        """Lưu báo cáo thống kê để đưa vào đồ án"""
        report_file = self.output_dir / "sampling_report_v2.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SAMPLING STRATEGY REPORT (THEORETICAL FRAMEWORK)\n")
            f.write("================================================\n\n")
            
            f.write("1. DATA DISTRIBUTION BY DELAY GROUP (QUOTA CHECK)\n")
            group_counts = df['DELAY_GROUP'].value_counts()
            total = len(df)
            for group, count in group_counts.items():
                f.write(f"   - {group}: {count} rows ({count/total:.1%})\n")
                
            f.write("\n2. COVERAGE CHECK\n")
            top_carrier_coverage = df['OP_CARRIER'].isin(self.top_carriers).mean()
            f.write(f"   - Top Carrier Presence: {top_carrier_coverage:.1%}\n")
            
            f.write("\n3. DETAILED LOG (MONTHLY)\n")
            f.write(f"{'Month':<6} | {'Group':<10} | {'Available':<10} | {'Sampled':<10}\n")
            f.write("-" * 45 + "\n")
            for s in stats:
                f.write(f"{s['Month']:<6} | {s['Group']:<10} | {s['Available']:<10} | {s['Sampled']:<10}\n")
        
        print(f"   - Report: {report_file}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Sửa chữ DA_AVD thành ADA cho đúng với máy của bạn
    INPUT_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/parquet/flights_weather"
    
    # Output giữ nguyên hoặc sửa lại tùy ý
    OUTPUT_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled"
    
    pipeline = FlightSamplingPipeline(INPUT_PATH, OUTPUT_PATH)
    pipeline.run_pipeline()