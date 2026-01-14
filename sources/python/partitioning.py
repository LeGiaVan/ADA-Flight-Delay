import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from pathlib import Path
import gc
from datetime import datetime
import time
import re
import json


def check_if_year_processed(output_dir, target_year):
    """Kiểm tra xem năm đã được xử lý chưa"""
    year_path = Path(output_dir) / f"year={target_year}"

    if year_path.exists():
        # Đếm số file parquet trong thư mục năm đó
        parquet_files = list(year_path.rglob("*.parquet"))
        if len(parquet_files) > 0:
            print(f"📊 Year {target_year} already has {len(parquet_files):,} parquet files")

            # Đếm tổng số rows
            total_rows = 0
            for pf in parquet_files[:5]:  # Chỉ check 5 file đầu
                try:
                    table = pq.read_table(pf)
                    total_rows += len(table)
                except:
                    pass

            if total_rows > 0:
                print(f"   Estimated {total_rows * (len(parquet_files) / 5):,.0f} rows")

            return True
    return False


def flight_weather_csv_to_parquet_incremental(csv_path, output_dir, chunk_size=500000, force_reprocess=False):
    """Process flight CSV incrementally - chỉ xử lý nếu chưa tồn tại"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Xác định năm từ filename
    year_filter = None
    # CHỈ XỬ LÝ NĂM 2023
    for year in [2023]:  # Chỉ kiểm tra năm 2023
        if str(year) in csv_path:
            year_filter = year
            break

    if year_filter is None:
        print(f"❌ Không tìm thấy năm 2023 trong filename: {csv_path}")
        return 0, 0, 0

    print(f"\n{'=' * 60}")
    print(f"🎯 Xử lý file: {Path(csv_path).name}")
    print(f"📅 Target year: {year_filter}")
    print(f"📁 Output: {output_dir}")
    print(f"{'=' * 60}")

    # Kiểm tra nếu năm đã được xử lý
    if not force_reprocess and check_if_year_processed(output_dir, year_filter):
        print(f"\n⏭️  SKIPPING: Năm {year_filter} đã được xử lý trước đó")
        print(f"   Sử dụng force_reprocess=True để xử lý lại")
        return 0, 0, 0

    # Schema với FL_DATE là string
    flight_weather_schema = {
        'FL_DATE': 'str',
        'OP_CARRIER': 'str',
        'OP_CARRIER_FL_NUM': 'int32',
        'ORIGIN': 'str',
        'DEST': 'str',
        'CRS_DEP_TIME': 'str',
        'DEP_TIME': 'str',
        'DEP_DELAY': 'float32',
        'TAXI_OUT': 'float32',
        'WHEELS_OFF': 'str',
        'WHEELS_ON': 'str',
        'TAXI_IN': 'float32',
        'CRS_ARR_TIME': 'str',
        'ARR_TIME': 'str',
        'ARR_DELAY': 'float32',
        'CRS_ELAPSED_TIME': 'float32',
        'ACTUAL_ELAPSED_TIME': 'float32',
        'AIR_TIME': 'float32',
        'FLIGHTS': 'int32',
        'MONTH': 'int32',
        'DAY_OF_MONTH': 'int32',
        'DAY_OF_WEEK': 'int32',
        'ORIGIN_INDEX': 'int32',
        'DEST_INDEX': 'int32',
        'O_TEMP': 'float32',
        'O_PRCP': 'float32',
        'O_WSPD': 'float32',
        'D_TEMP': 'float32',
        'D_PRCP': 'float32',
        'D_WSPD': 'float32',
        'O_LATITUDE': 'float32',
        'O_LONGITUDE': 'float32',
        'D_LATITUDE': 'float32',
        'D_LONGITUDE': 'float32'
    }

    chunk_number = 0
    total_rows = 0
    start_time = time.time()

    try:
        # Đếm tổng số chunks để ước tính
        print(f"\n📊 Đang đọc file CSV...")

        for chunk in pd.read_csv(csv_path,
                                 chunksize=chunk_size,
                                 dtype=flight_weather_schema,
                                 na_values=['NA', 'null', '', '\\N', 'NaN', 'nan'],
                                 low_memory=True,
                                 encoding='utf-8'):

            chunk_start_time = time.time()
            initial_rows = len(chunk)

            # 1. Basic cleaning
            essential_cols = ['ORIGIN', 'DEST', 'OP_CARRIER', 'FL_DATE']
            chunk_clean = chunk.dropna(subset=essential_cols)

            # 2. Extract year/month
            if len(chunk_clean) > 0:
                # Extract year từ FL_DATE string
                def extract_year(date_str):
                    if pd.isna(date_str):
                        return None
                    match = re.match(r'(\d{4})', str(date_str))
                    return int(match.group(1)) if match else None

                def extract_month(date_str):
                    if pd.isna(date_str):
                        return None
                    match = re.search(r'-(\d{1,2})-', str(date_str))
                    return int(match.group(1)) if match else None

                chunk_clean['year'] = chunk_clean['FL_DATE'].apply(extract_year)
                chunk_clean['month'] = chunk_clean['FL_DATE'].apply(extract_month)

                # Filter theo năm 2023
                chunk_clean = chunk_clean[chunk_clean['year'] == year_filter]
                chunk_clean = chunk_clean.dropna(subset=['year', 'month'])

                # Convert year/month to int
                chunk_clean['year'] = chunk_clean['year'].astype('int32')
                chunk_clean['month'] = chunk_clean['month'].astype('int32')

            # 3. Save by partition
            if len(chunk_clean) > 0:
                saved_in_chunk = 0

                for (year, month), month_group in chunk_clean.groupby(['year', 'month']):
                    if len(month_group) > 0:
                        partition_path = Path(output_dir) / f"year={year}" / f"month={month:02d}"
                        partition_path.mkdir(parents=True, exist_ok=True)

                        # Tạo tên file unique với timestamp để tránh overwrite
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = partition_path / f"chunk_{chunk_number:04d}_{timestamp}.parquet"

                        # Convert and save
                        table = pa.Table.from_pandas(month_group)
                        pq.write_table(table, output_file, compression='snappy')

                        file_size = output_file.stat().st_size / 1024 ** 2  # MB
                        saved_in_chunk += len(month_group)

                        del table

                total_rows += saved_in_chunk

            # 4. Cleanup
            del chunk_clean
            del chunk
            gc.collect()

            chunk_number += 1

            # Print progress mỗi 5 chunks
            if chunk_number % 5 == 0 or chunk_number == 1:
                elapsed = time.time() - start_time
                if elapsed > 0 and total_rows > 0:
                    speed = total_rows / elapsed
                    print(f"📈 Chunk {chunk_number:,}: {total_rows:,} rows "
                          f"({elapsed:.1f}s, {speed:.0f} rows/sec)")

        total_time = time.time() - start_time

        print(f"\n✅ HOÀN THÀNH: Năm {year_filter}")
        print(f"📊 Tổng rows: {total_rows:,}")
        print(f"⏱️  Thời gian: {total_time:.1f}s")
        print(f"🚀 Tốc độ: {total_rows / total_time:.0f} rows/s" if total_time > 0 else "")

        return total_rows, chunk_number, total_time

    except Exception as e:
        print(f"❌ Lỗi khi xử lý {csv_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0


def process_single_year(year=2023, force_reprocess=False):
    """Xử lý duy nhất năm 2023"""

    base_csv_dir = "D:/UEL/DA_AVD/ADA-Flight-Delay/data/raw/"
    base_output_dir = "D:/UEL/DA_AVD/ADA-Flight-Delay/data/parquet/flights_weather/"

    print("=" * 60)
    print(f"🚀 FLIGHT DATA PROCESSOR - YEAR {year}")
    print("=" * 60)
    print(f"Output sẽ được lưu tại: {base_output_dir}")
    print("-" * 60)

    # Tạo output directory
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    # Load progress từ metadata nếu có
    metadata_path = Path(base_output_dir) / "_PROCESSING_METADATA.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        processed_years = metadata.get('processed_years', [])
    else:
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'processed_years': [],
            'total_rows': 0,
            'file_sizes': {},
            'last_updated': datetime.now().isoformat()
        }
        processed_years = []

    csv_file = f"flight_with_weather_{year}.csv"
    csv_path = os.path.join(base_csv_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"❌ File không tồn tại: {csv_path}")
        return

    print(f"\n{'=' * 40}")
    print(f"🚀 Bắt đầu xử lý năm {year}")
    print(f"📁 File: {csv_file}")

    file_size_gb = os.path.getsize(csv_path) / (1024 ** 3)
    print(f"📏 Kích thước: {file_size_gb:.2f} GB")

    # Kiểm tra nếu đã xử lý và không force
    if year in processed_years and not force_reprocess:
        print(f"\n⏭️  Năm {year} đã được xử lý trước đó")
        check_output_structure(base_output_dir)
        return

    rows, chunks, proc_time = flight_weather_csv_to_parquet_incremental(
        csv_path=csv_path,
        output_dir=base_output_dir,
        chunk_size=500000,
        force_reprocess=force_reprocess
    )

    if rows > 0:
        # Cập nhật metadata
        if year not in metadata['processed_years']:
            metadata['processed_years'].append(year)
        
        # Tính tổng rows (chỉ cho năm 2023 trong trường hợp này)
        metadata['total_rows'] = rows
        metadata['file_sizes'][str(year)] = f"{file_size_gb:.2f}GB"
        metadata['last_updated'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Đã cập nhật metadata cho năm {year}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"🎉 XỬ LÝ HOÀN TẤT CHO NĂM {year}!")
    print(f"{'=' * 60}")
    print(f"📊 Tổng rows: {rows:,}")
    print(f"⏱️  Thời gian: {proc_time:.1f}s")
    print(f"🚀 Tốc độ: {rows / proc_time:.0f} rows/s" if proc_time > 0 else "")
    print(f"📁 Metadata saved to: {metadata_path}")
    
    # Check output structure
    print(f"\n📁 Kiểm tra cấu trúc output:")
    check_output_structure(base_output_dir)
    print(f"{'=' * 60}")


def check_output_structure(output_dir):
    """Check và hiển thị cấu trúc output"""
    output_base = Path(output_dir)
    if not output_base.exists():
        print("  ❌ Output directory không tồn tại!")
        return

    total_files = 0
    total_size_mb = 0
    total_rows_in_files = 0

    # CHỈ KIỂM TRA NĂM 2023
    for year_dir in sorted(output_base.glob("year=2023")):
        if year_dir.is_dir():
            year = year_dir.name.replace("year=", "")
            parquet_files = list(year_dir.rglob("*.parquet"))

            if parquet_files:
                year_size = 0
                year_rows = 0

                for parquet_file in parquet_files[:10]:  # Chỉ check 10 file đầu
                    try:
                        pf = pq.ParquetFile(parquet_file)
                        year_rows += pf.metadata.num_rows
                        year_size += parquet_file.stat().st_size
                    except Exception as e:
                        year_size += parquet_file.stat().st_size

                # Ước tính tổng
                estimated_rows = year_rows * (len(parquet_files) / min(10, len(parquet_files)))
                estimated_size_mb = (year_size * len(parquet_files) / min(10, len(parquet_files))) / (1024 ** 2)

                total_files += len(parquet_files)
                total_size_mb += estimated_size_mb
                total_rows_in_files += estimated_rows

                print(f"  📅 Year {year}: {len(parquet_files):,} files")
                print(f"    📊 Estimated: {estimated_rows:,.0f} rows, {estimated_size_mb:.1f} MB")

    if total_files > 0:
        print(f"\n📦 TỔNG: {total_files:,} parquet files")
        print(f"👥 Estimated {total_rows_in_files:,.0f} rows")
        print(f"💾 Estimated {total_size_mb:.1f} MB")
    else:
        print("  ⚠️ Không tìm thấy file parquet cho năm 2023!")


# Main execution
if __name__ == "__main__":
    # Chạy xử lý năm 2023
    # Có thể dùng force_reprocess=True để xử lý lại nếu cần
    process_single_year(year=2023, force_reprocess=False)