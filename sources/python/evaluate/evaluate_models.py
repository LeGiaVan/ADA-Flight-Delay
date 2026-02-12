import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. HÀM TÁI TẠO DỮ LIỆU TEST (Phải giống hệt logic Training)
# ==============================================================================
def prepare_test_data(csv_path):
    print("🔄 Đang tái tạo dữ liệu kiểm thử (Recreating Test Data)...")
    
    # Đọc dữ liệu
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {csv_path}")
        return None, None, None

    # --- BƯỚC 1: CLEANING (Giống Phase 1) ---
    cols_temp = ['ARR_DELAY', 'DEP_DELAY', 'CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 
                 'OP_CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK', 
                 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
    
    # Chỉ giữ cột cần thiết
    df = df[cols_temp].copy()
    
    # Fill NA thời tiết
    df[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']] = df[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']].fillna(0)
    
    # --- BƯỚC 2: FEATURE ENGINEERING (Giống Phase 2) ---
    
    # 2.1 Target Engineering
    df['ARR_DELAY'] = pd.to_numeric(df['ARR_DELAY'], errors='coerce').fillna(0)
    df['DEP_DELAY'] = pd.to_numeric(df['DEP_DELAY'], errors='coerce').fillna(0)
    df['TOTAL_DELAY'] = df['ARR_DELAY'] + df['DEP_DELAY']
    
    def classify_level(minutes):
        if minutes <= 0: return 0
        elif minutes <= 45: return 1
        elif minutes <= 90: return 2
        else: return 3
    
    df['DELAY_LEVEL'] = df['TOTAL_DELAY'].apply(classify_level)
    
    # 2.2 Input Engineering - Xử lý DateTime cho CRS_DEP_TIME
    try:
        # Chuyển sang datetime object để lấy giờ
        df['CRS_DEP_TIME_DT'] = pd.to_datetime(df['CRS_DEP_TIME'], errors='coerce')
        df['DEP_HOUR'] = df['CRS_DEP_TIME_DT'].dt.hour
        df['DEP_HOUR'] = df['DEP_HOUR'].fillna(12).astype(int) # Fallback nếu lỗi
    except:
        df['DEP_HOUR'] = 12
        
    # Cuối tuần
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)
    
    # Mùa vụ
    def get_season(month):
        try:
            m = int(month)
            if m in [3, 4, 5]: return 'Spring'
            elif m in [6, 7, 8]: return 'Summer'
            elif m in [9, 10, 11]: return 'Fall'
            else: return 'Winter'
        except: return 'Unknown'
    df['SEASON'] = df['MONTH'].apply(get_season)
    
    # Route Risk (Tính lại trên toàn bộ tập dữ liệu để khớp logic train cũ)
    route_stats = df.groupby(['ORIGIN', 'DEST'])['TOTAL_DELAY'].mean().reset_index()
    route_stats.rename(columns={'TOTAL_DELAY': 'ROUTE_RISK'}, inplace=True)
    df = df.merge(route_stats, on=['ORIGIN', 'DEST'], how='left')
    global_mean = df['TOTAL_DELAY'].mean()
    df['ROUTE_RISK'] = df['ROUTE_RISK'].fillna(global_mean)

    # Gom nhóm Origin/Dest
    df['ORIGIN'] = df['ORIGIN'].astype(str)
    df['DEST'] = df['DEST'].astype(str)
    
    top_20_origin = df['ORIGIN'].value_counts().nlargest(20).index
    df['ORIGIN_GROUPED'] = df['ORIGIN'].apply(lambda x: x if x in top_20_origin else 'OTHER')
    
    top_20_dest = df['DEST'].value_counts().nlargest(20).index
    df['DEST_GROUPED'] = df['DEST'].apply(lambda x: x if x in top_20_dest else 'OTHER')
    
    # --- BƯỚC 3: CHUẨN BỊ X, Y ---
    cols_X = ['CRS_ELAPSED_TIME', 'ROUTE_RISK', 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD',
              'OP_CARRIER', 'ORIGIN_GROUPED', 'DEST_GROUPED', 'DEP_HOUR', 'MONTH', 'SEASON', 'IS_WEEKEND']
    
    X = df[cols_X]
    y_reg = df['TOTAL_DELAY']
    y_clf = df['DELAY_LEVEL']
    
    # --- BƯỚC 4: SPLIT (Quan trọng: random_state=42 để khớp Training) ---
    _, X_test, _, y_reg_test, _, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    print(f"   -> Đã tạo xong tập Test với {len(X_test)} mẫu.")
    return X_test, y_reg_test, y_clf_test

# ==============================================================================
# 2. HÀM ĐÁNH GIÁ TỪNG MODEL
# ==============================================================================
def evaluate_model(name, model_path, X_test, y_true_clf, y_true_reg=None):
    print(f"\n{'='*60}")
    print(f"🧪 ĐANG ĐÁNH GIÁ: {name}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return None

    # Dự báo
    y_pred_raw = model.predict(X_test)
    
    # Xử lý riêng cho Linear Regression
    if name == "Linear Regression":
        if y_true_reg is not None:
            rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_raw))
            r2 = r2_score(y_true_reg, y_pred_raw)
            print(f"   📊 Regression Metrics:")
            print(f"      - RMSE: {rmse:.2f} phút")
            print(f"      - R2:   {r2:.4f}")
        
        # Convert số phút sang Class
        def to_class(m):
            if m <= 0: return 0
            elif m <= 45: return 1
            elif m <= 90: return 2
            else: return 3
        y_pred_class = [to_class(m) for m in y_pred_raw]
        
    else:
        # Logistic & XGBoost ra thẳng Class
        y_pred_class = y_pred_raw

    # In báo cáo
    target_names = ['Good (<=0)', 'Minor (0-45)', 'Moderate (45-90)', 'Severe (>90)']
    print("\n   📋 Classification Report:")
    print(classification_report(y_true_clf, y_pred_class, target_names=target_names))
    
    return y_pred_class

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Cấu hình đường dẫn (Sửa lại nếu cần)
    CSV_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    MODEL_DIR = "models"
    REPORT_DIR = "data/report"
    
    if not os.path.exists(REPORT_DIR): os.makedirs(REPORT_DIR)

    # 1. Tái tạo dữ liệu Test
    X_test, y_reg_test, y_clf_test = prepare_test_data(CSV_PATH)
    
    if X_test is not None:
        # 2. Danh sách model cần đánh giá
        models_config = [
            ("Linear Regression", "flight_delay_linear.pkl", True),
            ("Logistic Regression", "flight_delay_logistic.pkl", False),
            ("XGBoost Classifier", "flight_delay_xgboost.pkl", False)
        ]
        
        predictions = {}
        
        # 3. Chạy vòng lặp đánh giá
        for name, filename, is_reg in models_config:
            path = os.path.join(MODEL_DIR, filename)
            # Truyền y_reg_test nếu là Linear, ngược lại None
            y_true_r = y_reg_test if is_reg else None
            
            pred = evaluate_model(name, path, X_test, y_clf_test, y_true_r)
            if pred is not None:
                predictions[name] = pred

        # 4. Vẽ Confusion Matrix so sánh
        if predictions:
            print("\n🎨 Đang vẽ biểu đồ so sánh...")
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            class_labels = ['Good', 'Minor', 'Moderate', 'Severe']
            
            # Nếu chỉ có 1 hoặc 2 model thành công thì xử lý axes cho phù hợp
            if len(predictions) < 3:
                axes = [axes] if len(predictions) == 1 else axes
            
            for ax, (name, pred) in zip(np.ravel(axes), predictions.items()):
                cm = confusion_matrix(y_clf_test, pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_labels, yticklabels=class_labels, ax=ax)
                ax.set_title(f"{name}")
                ax.set_xlabel("Dự báo (Predicted)")
                ax.set_ylabel("Thực tế (Actual)")
            
            plt.tight_layout()
            save_path = os.path.join(REPORT_DIR, "evaluation_comparison.png")
            plt.savefig(save_path)
            print(f"✅ Đã lưu biểu đồ tại: {save_path}")
            plt.show()