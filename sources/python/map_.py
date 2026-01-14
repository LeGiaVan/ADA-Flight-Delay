import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_simple_airport_map(csv_path, output_image='us_airports_map.png'):
    """
    Tạo bản đồ sân bay đơn giản với matplotlib
    """
    
    print("🗺️ ĐANG TẠO BẢN ĐỒ SÂN BAY...")
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(csv_path)
    print(f"✅ Đã đọc {len(df):,} dòng")
    
    # 2. Chuẩn bị dữ liệu sân bay
    origin_airports = df[['ORIGIN', 'O_LATITUDE', 'O_LONGITUDE']].copy()
    origin_airports.columns = ['airport_code', 'latitude', 'longitude']
    
    dest_airports = df[['DEST', 'D_LATITUDE', 'D_LONGITUDE']].copy()
    dest_airports.columns = ['airport_code', 'latitude', 'longitude']
    
    all_airports = pd.concat([origin_airports, dest_airports], ignore_index=True)
    all_airports = all_airports.dropna(subset=['latitude', 'longitude'])
    
    # 3. Thống kê sân bay
    airport_stats = all_airports.groupby('airport_code').agg({
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    airport_counts = all_airports['airport_code'].value_counts().reset_index()
    airport_counts.columns = ['airport_code', 'total_flights']
    
    airport_stats = pd.merge(airport_stats, airport_counts, on='airport_code')
    airport_stats = airport_stats.sort_values('total_flights', ascending=False)
    
    print(f"✈️ Tổng số sân bay: {len(airport_stats)}")
    
    # 4. Tạo bản đồ
    plt.figure(figsize=(15, 10))
    
    # Vẽ outline nước Mỹ (đơn giản)
    # Tọa độ bounding box của Mỹ
    usa_lon_min, usa_lon_max = -125, -66
    usa_lat_min, usa_lat_max = 24, 50
    
    # Tạo background
    plt.xlim(usa_lon_min, usa_lon_max)
    plt.ylim(usa_lat_min, usa_lat_max)
    plt.gca().set_facecolor('#e8f4f8')  # Màu nền xanh nhạt
    
    # Vẽ sân bay
    sizes = np.sqrt(airport_stats['total_flights']) * 3  # Scale size
    
    scatter = plt.scatter(
        airport_stats['longitude'],
        airport_stats['latitude'],
        s=sizes,
        c=airport_stats['total_flights'],
        cmap='RdYlBu_r',  # Red-Yellow-Blue reversed
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Thêm colorbar
    plt.colorbar(scatter, label='Số chuyến bay')
    
    # Thêm labels cho top airports
    top_n = 15
    for i, row in airport_stats.head(top_n).iterrows():
        plt.annotate(
            row['airport_code'],
            xy=(row['longitude'], row['latitude']),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkred'
        )
    
    # Thêm title và labels
    plt.title('BẢN ĐỒ SÂN BAY MỸ - NĂM 2023', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Kinh độ (Longitude)', fontsize=12)
    plt.ylabel('Vĩ độ (Latitude)', fontsize=12)
    
    # Thêm grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Thêm chú thích
    plt.figtext(0.5, 0.01, 
                f'✈️ Tổng {len(airport_stats)} sân bay | 📊 {airport_stats["total_flights"].sum():,} chuyến bay',
                ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    
    # 5. Lưu hình ảnh
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"/n✅ ĐÃ LƯU BẢN ĐỒ: {output_image}")
    
    # 6. Hiển thị thống kê
    print("/n🏆 TOP 15 SÂN BAY NHIỀU CHUYẾN NHẤT:")
    print("-" * 60)
    for i, row in airport_stats.head(15).iterrows():
        print(f"{i+1:2d}. {row['airport_code']}: {row['total_flights']:4d} chuyến "
              f"(lat: {row['latitude']:.2f}, lon: {row['longitude']:.2f})")
    
    # 7. Lưu thống kê CSV
    stats_csv = output_image.replace('.png', '_stats.csv')
    airport_stats.to_csv(stats_csv, index=False, encoding='utf-8')
    print(f"/n📊 Đã lưu thống kê: {stats_csv}")
    
    # Hiển thị bản đồ
    plt.show()
    
    return airport_stats

# Chạy chương trình
if __name__ == "__main__":
    CSV_PATH = 'D:/UEL/DA_AVD/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv'
    OUTPUT_IMAGE = 'D:/UEL/DA_AVD/ADA-Flight-Delay/data/visualization/us_airports_map_2023.png'
    
    try:
        airport_stats = create_simple_airport_map(CSV_PATH, OUTPUT_IMAGE)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {CSV_PATH}")
        print("   Vui lòng kiểm tra đường dẫn!")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")