# map_simple.py - Dùng matplotlib (KHÔNG cần cài thêm)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def create_airport_map_matplotlib(csv_path, output_dir='visualization'):
    """
    Tạo bản đồ sân bay đơn giản với matplotlib
    Không cần cài thêm thư viện!
    """
    
    print("=" * 60)
    print("🗺️ TẠO BẢN ĐỒ SÂN BAY VỚI MATPLOTLIB")
    print("=" * 60)
    
    # 1. Kiểm tra file tồn tại
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ KHÔNG TÌM THẤY FILE: {csv_path}")
        print(f"   Thư mục hiện tại: {os.getcwd()}")
        print(f"   Files trong thư mục data/sampled/:")
        data_dir = Path("data/sampled")
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                print(f"   - {f.name}")
        return None
    
    print(f"📖 Đọc file: {csv_path}")
    
    # 2. Đọc dữ liệu
    df = pd.read_csv(csv_path)
    print(f"✅ Đã đọc {len(df):,} dòng, {len(df.columns)} cột")
    
    # 3. Xử lý dữ liệu sân bay
    print("\n🔍 XỬ LÝ DỮ LIỆU SÂN BAY...")
    
    # Lấy sân bay xuất phát
    origins = df[['ORIGIN', 'O_LATITUDE', 'O_LONGITUDE']].copy()
    origins.columns = ['code', 'lat', 'lon']
    origins = origins.dropna()
    
    # Lấy sân bay đến
    dests = df[['DEST', 'D_LATITUDE', 'D_LONGITUDE']].copy()
    dests.columns = ['code', 'lat', 'lon']
    dests = dests.dropna()
    
    # Kết hợp
    all_airports = pd.concat([origins, dests], ignore_index=True)
    
    # Thống kê
    airport_stats = all_airports.groupby('code').agg({
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    flight_counts = all_airports['code'].value_counts().reset_index()
    flight_counts.columns = ['code', 'total_flights']
    
    airport_stats = pd.merge(airport_stats, flight_counts, on='code')
    airport_stats = airport_stats.sort_values('total_flights', ascending=False)
    
    print(f"✈️ Số sân bay duy nhất: {len(airport_stats)}")
    print(f"📊 Tổng số chuyến bay: {airport_stats['total_flights'].sum():,}")
    
    # 4. Hiển thị top sân bay
    print("\n🏆 TOP 15 SÂN BAY NHIỀU CHUYẾN NHẤT:")
    print("-" * 50)
    for i, (_, row) in enumerate(airport_stats.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['code']}: {row['total_flights']:4d} chuyến")
    
    # 5. Tạo thư mục output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. Tạo bản đồ
    print("\n🎨 ĐANG VẼ BẢN ĐỒ...")
    
    plt.figure(figsize=(16, 10))
    
    # Giới hạn bản đồ nước Mỹ
    plt.xlim(-130, -65)  # Longitude
    plt.ylim(20, 55)     # Latitude
    
    # Màu nền
    plt.gca().set_facecolor('#f0f8ff')
    
    # Tính kích thước marker
    sizes = np.sqrt(airport_stats['total_flights']) * 8
    
    # Vẽ các sân bay
    scatter = plt.scatter(
        airport_stats['lon'],
        airport_stats['lat'],
        s=sizes,
        c=airport_stats['total_flights'],
        cmap='YlOrRd',  # Yellow-Orange-Red
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        zorder=2
    )
    
    # Thêm colorbar
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Số chuyến bay', fontsize=12)
    
    # Thêm label cho top 10 sân bay
    top_10 = airport_stats.head(10)
    for _, row in top_10.iterrows():
        plt.annotate(
            row['code'],
            xy=(row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    # Tiêu đề và labels
    plt.title('BẢN ĐỒ PHÂN BỐ SÂN BAY MỸ - 2023', 
              fontsize=18, fontweight='bold', pad=20, color='darkblue')
    
    plt.xlabel('Kinh độ (Longitude)', fontsize=12)
    plt.ylabel('Vĩ độ (Latitude)', fontsize=12)
    
    # Grid
    plt.grid(True, alpha=0.2, linestyle='--', zorder=1)
    
    # Chú thích về kích thước
    plt.figtext(0.5, 0.01, 
                f'Kích thước điểm ∝ √(số chuyến bay) | Tổng: {len(airport_stats)} sân bay, {airport_stats["total_flights"].sum():,} chuyến bay',
                ha='center', fontsize=11, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 7. Lưu file
    output_png = output_dir / 'us_airports_map_2023.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ ĐÃ LƯU BẢN ĐỒ: {output_png}")
    
    # 8. Lưu thống kê CSV
    stats_csv = output_dir / 'airport_statistics_2023.csv'
    airport_stats.to_csv(stats_csv, index=False, encoding='utf-8')
    print(f"📊 Đã lưu thống kê: {stats_csv}")
    
    # 9. Hiển thị bản đồ
    plt.show()
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH! Bản đồ đã được hiển thị và lưu.")
    print("=" * 60)
    
    return airport_stats

def display_detailed_stats(airport_stats):
    """Hiển thị thống kê chi tiết"""
    
    print("\n📋 THỐNG KÊ CHI TIẾT SÂN BAY")
    print("=" * 60)
    
    total_flights = airport_stats['total_flights'].sum()
    avg_flights = airport_stats['total_flights'].mean()
    
    print(f"📈 Tổng số chuyến bay: {total_flights:,}")
    print(f"📊 Trung bình: {avg_flights:.1f} chuyến/sân bay")
    print(f"📉 Median: {airport_stats['total_flights'].median():.1f} chuyến")
    print(f"🔥 Max: {airport_stats['total_flights'].max()} chuyến ({airport_stats.iloc[0]['code']})")
    print(f"❄️  Min: {airport_stats['total_flights'].min()} chuyến")
    
    # Phân nhóm
    print("\n📊 PHÂN NHÓM SÂN BAY THEO SỐ CHUYẾN:")
    bins = [0, 10, 50, 100, 200, 500, 1000]
    labels = ['Rất ít (0-10)', 'Ít (11-50)', 'Trung bình (51-100)', 
              'Nhiều (101-200)', 'Rất nhiều (201-500)', 'Cực nhiều (500+)']
    
    airport_stats['group'] = pd.cut(airport_stats['total_flights'], bins=bins, labels=labels)
    group_counts = airport_stats['group'].value_counts().sort_index()
    
    for group, count in group_counts.items():
        percentage = (count / len(airport_stats)) * 100
        print(f"  - {group}: {count:3d} sân bay ({percentage:5.1f}%)")

# CHẠY CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    # ĐƯỜNG DẪN FILE CSV - ĐIỀU CHỈNH THEO THỰC TẾ
    CSV_PATHS = [
        "D:/UEL/DA_AVD/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv",
        "data/sampled/flight_data_sampled_2023.csv",
        "../data/sampled/flight_data_sampled_2023.csv",
        "./flight_data_sampled_2023.csv"
    ]
    
    # Thử các đường dẫn khác nhau
    csv_found = None
    for csv_path in CSV_PATHS:
        if Path(csv_path).exists():
            csv_found = csv_path
            break
    
    if csv_found:
        print(f"📁 Tìm thấy file tại: {csv_found}")
        stats = create_airport_map_matplotlib(csv_found, output_dir="data/visualization")
        if stats is not None:
            display_detailed_stats(stats)
    else:
        print("❌ KHÔNG TÌM THẤY FILE CSV!")
        print("\nHãy kiểm tra:")
        print("1. File có tồn tại không?")
        print("2. Đường dẫn đúng không?")
        print("3. Chạy trong thư mục D:/UEL/DA_AVD/ADA-Flight-Delay/")
        print("\nThử chạy:")
        print("cd D:/UEL/DA_AVD/ADA-Flight-Delay")
        print("python sources/python/map_simple.py")