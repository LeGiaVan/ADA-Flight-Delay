import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# ==============================================================================
# PHASE 1: DATA CLEANING & SELECTION
# ==============================================================================
def phase_1_data_cleaning(df):
    print("🔹 PHASE 1: DATA CLEANING & SELECTION...")
    
    # 1. Giữ lại các cột cần thiết
    cols_to_keep_temp = [
        'ARR_DELAY', 'DEP_DELAY',     # Để tính Y
        'CRS_DEP_TIME',               # Để tính Giờ bay
        'CRS_ELAPSED_TIME',           # Dự kiến bay bao lâu
        'OP_CARRIER', 'ORIGIN', 'DEST',
        'MONTH', 'DAY_OF_WEEK',
        'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD'
    ]
    
    # Lọc cột
    df_clean = df[cols_to_keep_temp].copy()
    
    # 2. Xử lý Missing Value cơ bản
    df_clean[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']] = df_clean[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']].fillna(0)
    
    # Drop dòng thiếu Delay
    df_clean = df_clean.dropna(subset=['ARR_DELAY', 'DEP_DELAY'])
    
    print(f"   -> Đã loại bỏ hoàn toàn các biến Future Leakage.")
    print(f"   -> Kích thước dữ liệu sạch: {df_clean.shape}")
    
    return df_clean

# ==============================================================================
# PHASE 2: FEATURE ENGINEERING
# ==============================================================================
def phase_2_feature_engineering(df):
    print("🔹 PHASE 2: FEATURE ENGINEERING...")
    
    # 1. TẠO BIẾN MỤC TIÊU (TARGET Y)
    df['ARR_DELAY'] = pd.to_numeric(df['ARR_DELAY'], errors='coerce').fillna(0)
    df['DEP_DELAY'] = pd.to_numeric(df['DEP_DELAY'], errors='coerce').fillna(0)
    
    df['TOTAL_DELAY'] = df['ARR_DELAY'] + df['DEP_DELAY']
    
    def classify_level(minutes):
        if minutes <= 0: return 0
        elif minutes <= 45: return 1
        elif minutes <= 90: return 2
        else: return 3
        
    df['DELAY_LEVEL'] = df['TOTAL_DELAY'].apply(classify_level)
    
    # 2. TẠO BIẾN ĐẦU VÀO (INPUT X)
    
    # A. Giờ bay (DEP_HOUR) - Xử lý chuẩn DateTime
    try:
        df['CRS_DEP_TIME_DT'] = pd.to_datetime(df['CRS_DEP_TIME'], errors='coerce')
        df['DEP_HOUR'] = df['CRS_DEP_TIME_DT'].dt.hour
        df['DEP_HOUR'] = df['DEP_HOUR'].fillna(12).astype(int) # Fallback 12h trưa
    except:
        df['DEP_HOUR'] = 12

    # B. Cuối tuần (IS_WEEKEND)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)
    
    # C. Mùa vụ (SEASON)
    def get_season(month):
        try:
            m = int(month)
            if m in [3, 4, 5]: return 'Spring'
            elif m in [6, 7, 8]: return 'Summer'
            elif m in [9, 10, 11]: return 'Fall'
            else: return 'Winter'
        except: return 'Unknown'
        
    df['SEASON'] = df['MONTH'].apply(get_season)
    
    # D. ROUTE_RISK (Target Encoding)
    route_stats = df.groupby(['ORIGIN', 'DEST'])['TOTAL_DELAY'].mean().reset_index()
    route_stats.rename(columns={'TOTAL_DELAY': 'ROUTE_RISK'}, inplace=True)
    
    df = df.merge(route_stats, on=['ORIGIN', 'DEST'], how='left')
    global_mean = df['TOTAL_DELAY'].mean()
    df['ROUTE_RISK'] = df['ROUTE_RISK'].fillna(global_mean) 
    
    # E. Gom nhóm ORIGIN/DEST
    df['ORIGIN'] = df['ORIGIN'].astype(str)
    df['DEST'] = df['DEST'].astype(str)
    
    top_20_origin = df['ORIGIN'].value_counts().nlargest(20).index
    df['ORIGIN_GROUPED'] = df['ORIGIN'].apply(lambda x: x if x in top_20_origin else 'OTHER')
    
    top_20_dest = df['DEST'].value_counts().nlargest(20).index
    df['DEST_GROUPED'] = df['DEST'].apply(lambda x: x if x in top_20_dest else 'OTHER')
    
    # 3. DROP CÁC CỘT KHÔNG DÙNG NỮA
    cols_to_drop = ['ARR_DELAY', 'DEP_DELAY', 'CRS_DEP_TIME', 'CRS_DEP_TIME_DT', 'ORIGIN', 'DEST']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    df_final = df.drop(columns=cols_to_drop)
    
    print(f"   -> Hoàn thành Phase 2. Kích thước cuối cùng: {df_final.shape}")
    return df_final

# ==============================================================================
# PHASE 3: DATA PREPROCESSING
# ==============================================================================
def phase_3_preprocessing(df):
    print("🔹 PHASE 3: DATA PREPROCESSING...")
    
    # 1. Tách X và Y
    target_reg = 'TOTAL_DELAY'
    target_clf = 'DELAY_LEVEL'
    
    X = df.drop(columns=[target_reg, target_clf])
    y_reg = df[target_reg]
    y_clf = df[target_clf]
    
    # 2. Định nghĩa các nhóm biến
    numeric_features = ['CRS_ELAPSED_TIME', 'ROUTE_RISK', 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
    categorical_features = ['OP_CARRIER', 'ORIGIN_GROUPED', 'DEST_GROUPED', 'DEP_HOUR', 'MONTH', 'SEASON', 'IS_WEEKEND']
    
    # 3. Tạo Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # 4. Chia tập Train/Test
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    print(f"   -> Train shape: {X_train.shape}")
    print(f"   -> Test shape:  {X_test.shape}")
    
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor

# ==============================================================================
# PHASE 4: MODELING, REPORTING & SAVING
# ==============================================================================
def phase_4_modeling_report_save(X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, save_dir='models', report_dir='data/report'):
    print("🔹 PHASE 4: MODELING, REPORTING & SAVING...")
    
    # Tạo thư mục
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    target_names = ['Good (<=0)', 'Minor (0-45)', 'Moderate (45-90)', 'Severe (>90)']
    
    # =========================================================
    # MODEL 1: LINEAR REGRESSION
    # =========================================================
    print(f"\n1️⃣  LINEAR REGRESSION...")
    
    model_lin = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])
    model_lin.fit(X_train, y_reg_train)
    
    # Đánh giá
    y_pred_lin = model_lin.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_lin))
    r2 = r2_score(y_reg_test, y_pred_lin)
    
    # Chuyển đổi sang Class để tính Accuracy so sánh
    def to_class(m):
        if m <= 0: return 0
        elif m <= 45: return 1
        elif m <= 90: return 2
        else: return 3
    y_pred_lin_class = [to_class(m) for m in y_pred_lin]
    acc_lin = accuracy_score(y_clf_test, y_pred_lin_class)
    
    # In ra console
    print(f"   -> RMSE: {rmse:.2f}")
    print(f"   -> R2: {r2:.4f}")
    
    # Ghi file report
    with open(os.path.join(report_dir, 'linear_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== LINEAR REGRESSION REPORT ===\n")
        f.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("CONVERTED TO CLASSIFICATION METRICS:\n")
        f.write(f"Accuracy (Converted): {acc_lin:.4f}\n\n")
        f.write("Detailed Classification Report (Converted):\n")
        f.write(classification_report(y_clf_test, y_pred_lin_class, target_names=target_names))

    # Lưu model
    joblib.dump(model_lin, os.path.join(save_dir, 'flight_delay_linear.pkl'))

    # =========================================================
    # MODEL 2: LOGISTIC REGRESSION
    # =========================================================
    print(f"\n2️⃣  LOGISTIC REGRESSION...")
    
    model_log = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression(
                                    max_iter=2000,
                                    class_weight='balanced',
                                    solver='lbfgs'
                                ))])
    model_log.fit(X_train, y_clf_train)
    
    # Đánh giá
    y_pred_log = model_log.predict(X_test)
    report_log = classification_report(y_clf_test, y_pred_log, target_names=target_names)
    cm_log = confusion_matrix(y_clf_test, y_pred_log)
    
    print(f"   -> Accuracy: {accuracy_score(y_clf_test, y_pred_log):.2%}")
    
    # Ghi file report
    with open(os.path.join(report_dir, 'logistic_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== LOGISTIC REGRESSION REPORT ===\n")
        f.write(f"Accuracy: {accuracy_score(y_clf_test, y_pred_log):.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_log)
        f.write("\nCONFUSION MATRIX:\n")
        f.write(str(cm_log))
        
    # Lưu model
    joblib.dump(model_log, os.path.join(save_dir, 'flight_delay_logistic.pkl'))

    # =========================================================
    # MODEL 3: XGBOOST CLASSIFIER
    # =========================================================
    print(f"\n3️⃣  XGBOOST CLASSIFIER...")
    
    # Xử lý dữ liệu & Weight
    X_train_proc = preprocessor.fit_transform(X_train, y_clf_train)
    X_test_proc = preprocessor.transform(X_test)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_clf_train)
    
    xgb_clf = XGBClassifier(
        objective='multi:softmax', 
        num_class=4, 
        eval_metric='mlogloss',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        random_state=42
    )
    
    xgb_clf.fit(X_train_proc, y_clf_train, sample_weight=sample_weights)
    
    # Đánh giá
    y_pred_xgb = xgb_clf.predict(X_test_proc)
    report_xgb = classification_report(y_clf_test, y_pred_xgb, target_names=target_names)
    cm_xgb = confusion_matrix(y_clf_test, y_pred_xgb)
    
    print(f"   -> Accuracy: {accuracy_score(y_clf_test, y_pred_xgb):.2%}")
    
    # Ghi file report
    with open(os.path.join(report_dir, 'xgboost_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== XGBOOST CLASSIFIER REPORT ===\n")
        f.write(f"Accuracy: {accuracy_score(y_clf_test, y_pred_xgb):.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_xgb)
        f.write("\nCONFUSION MATRIX:\n")
        f.write(str(cm_xgb))
        
        # Thêm Top Features
        try:
            f.write("\n\nTOP 10 FEATURES IMPORTANCE:\n")
            ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()
            num_cols = ['CRS_ELAPSED_TIME', 'ROUTE_RISK', 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
            all_cols = np.concatenate([num_cols, ohe_cols])
            importances = xgb_clf.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            for i in range(10):
                f.write(f"{i+1}. {all_cols[indices[i]]}: {importances[indices[i]]:.4f}\n")
        except:
            pass
            
    # Lưu model
    full_xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf)
    ])
    joblib.dump(full_xgb_pipeline, os.path.join(save_dir, 'flight_delay_xgboost.pkl'))

    print(f"\n✅ Đã xuất 3 báo cáo chi tiết vào thư mục: {report_dir}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    CSV_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    
    try:
        df_raw = pd.read_csv(CSV_PATH)
        
        df_p1 = phase_1_data_cleaning(df_raw)
        df_p2 = phase_2_feature_engineering(df_p1)
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, preproc = phase_3_preprocessing(df_p2)
        
        phase_4_modeling_report_save(X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, preproc)
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {CSV_PATH}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()