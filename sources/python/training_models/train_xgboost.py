import pandas as pd
import numpy as np
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# --- PHASE 1, 2, 3 (GIỮ NGUYÊN) ---
def phase_1_data_cleaning(df):
    cols = ['ARR_DELAY', 'DEP_DELAY', 'CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 
            'OP_CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK', 
            'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
    cols_to_keep = [c for c in cols if c in df.columns]
    df_clean = df[cols_to_keep].copy()
    df_clean[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']] = df_clean[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']].fillna(0)
    return df_clean.dropna(subset=['ARR_DELAY', 'DEP_DELAY'])

def phase_2_feature_engineering(df):
    df['ARR_DELAY'] = pd.to_numeric(df['ARR_DELAY'], errors='coerce').fillna(0)
    df['DEP_DELAY'] = pd.to_numeric(df['DEP_DELAY'], errors='coerce').fillna(0)
    df['TOTAL_DELAY'] = df['ARR_DELAY'] + df['DEP_DELAY']
    df['DELAY_LEVEL'] = df['TOTAL_DELAY'].apply(lambda x: 0 if x<=0 else (1 if x<=45 else (2 if x<=90 else 3)))
    try:
        df['CRS_DEP_TIME_DT'] = pd.to_datetime(df['CRS_DEP_TIME'], errors='coerce')
        df['DEP_HOUR'] = df['CRS_DEP_TIME_DT'].dt.hour.fillna(12).astype(int)
    except: df['DEP_HOUR'] = 12
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)
    def get_season(m):
        try: return {3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall'}.get(int(m), 'Winter')
        except: return 'Unknown'
    df['SEASON'] = df['MONTH'].apply(get_season)
    route_stats = df.groupby(['ORIGIN', 'DEST'])['TOTAL_DELAY'].mean().reset_index().rename(columns={'TOTAL_DELAY': 'ROUTE_RISK'})
    df = df.merge(route_stats, on=['ORIGIN', 'DEST'], how='left')
    df['ROUTE_RISK'] = df['ROUTE_RISK'].fillna(df['TOTAL_DELAY'].mean())
    df['ORIGIN'] = df['ORIGIN'].astype(str)
    df['DEST'] = df['DEST'].astype(str)
    top_20_org = df['ORIGIN'].value_counts().nlargest(20).index
    df['ORIGIN_GROUPED'] = df['ORIGIN'].apply(lambda x: x if x in top_20_org else 'OTHER')
    top_20_dst = df['DEST'].value_counts().nlargest(20).index
    df['DEST_GROUPED'] = df['DEST'].apply(lambda x: x if x in top_20_dst else 'OTHER')
    return df.drop(columns=['ARR_DELAY', 'DEP_DELAY', 'CRS_DEP_TIME', 'CRS_DEP_TIME_DT', 'ORIGIN', 'DEST'], errors='ignore')

def phase_3_preprocessing(df):
    X = df.drop(columns=['TOTAL_DELAY', 'DELAY_LEVEL'])
    y_reg = df['TOTAL_DELAY']
    y_clf = df['DELAY_LEVEL']
    num_cols = ['CRS_ELAPSED_TIME', 'ROUTE_RISK', 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
    cat_cols = ['OP_CARRIER', 'ORIGIN_GROUPED', 'DEST_GROUPED', 'DEP_HOUR', 'MONTH', 'SEASON', 'IS_WEEKEND']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor

# --- PHASE 4: MANUAL TUNING & REPORTING ---
def train_xgboost_detailed(X_train, X_test, y_train, y_test, preprocessor):
    print("\n🚀 TRAINING XGBOOST (DETAILED PER-CLASS REPORT)...")
    save_dir, report_dir = 'models', 'data/report'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    # Định nghĩa Grid tham số để chạy thử
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1]
    }
    
    print("   ⚙️ Preprocessing data and Calculating Weights...")
    # Fit preprocessor một lần để dùng lại cho nhanh
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Tính sample weight (Quan trọng cho dữ liệu mất cân bằng)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    report_path = os.path.join(report_dir, 'xgboost_detailed_report.txt')
    best_score = -1
    best_model = None
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header bảng báo cáo
        f.write("=== XGBOOST: PER-CLASS PERFORMANCE ===\n\n")
        header = (f"{'N_Est':<6} | {'Depth':<6} | {'LR':<6} || "
                  f"{'GOOD (0)':<12} | {'MINOR (1)':<12} | {'MODERATE (2)':<12} | {'SEVERE (3)':<12}\n")
        sub_header = (f"{'':<23}|| "
                      f"{'P':<5} {'R':<5} | {'P':<5} {'R':<5} | {'P':<5} {'R':<5} | {'P':<5} {'R':<5}\n")
        
        f.write(header)
        f.write(sub_header)
        f.write("-" * 110 + "\n")
        
        print(f"   ⚙️ Đang chạy vòng lặp tham số...")
        
        # Duyệt qua từng tổ hợp tham số
        for params in ParameterGrid(param_grid):
            # In ra màn hình để bạn biết nó đang chạy đến đâu
            print(f"   -> Testing N={params['n_estimators']}, Depth={params['max_depth']}, LR={params['learning_rate']}...", end="")
            
            try:
                clf = XGBClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    objective='multi:softmax', 
                    num_class=4, 
                    eval_metric='mlogloss',
                    n_jobs=-1,
                    random_state=42
                )
                
                # Train model với weights
                clf.fit(X_train_proc, y_train, sample_weight=sample_weights)
                y_pred = clf.predict(X_test_proc)
                
                # Lấy report chi tiết
                report = classification_report(y_test, y_pred, labels=[0,1,2,3], output_dict=True)
                
                # Trích xuất Precision (P) và Recall (R)
                p0, r0 = report['0']['precision'], report['0']['recall']
                p1, r1 = report['1']['precision'], report['1']['recall']
                p2, r2 = report['2']['precision'], report['2']['recall']
                p3, r3 = report['3']['precision'], report['3']['recall']
                
                # Ghi dòng dữ liệu vào file
                row = (f"{params['n_estimators']:<6} | {params['max_depth']:<6} | {params['learning_rate']:<6} || "
                       f"{p0:.2f}  {r0:.2f}  | {p1:.2f}  {r1:.2f}  | {p2:.2f}  {r2:.2f}  | {p3:.2f}  {r3:.2f}\n")
                f.write(row)
                print(" OK")
                
                # Lưu lại model có Severe Recall (r3) cao nhất
                if r3 > best_score:
                    best_score = r3
                    best_model = clf
                    
            except Exception as e:
                print(f" LỖI: {e}")
                f.write(f"{params['n_estimators']:<6} | ERROR: {str(e)}\n")
        
        f.write("-" * 110 + "\n")
        if best_model:
            f.write(f"Best SEVERE RECALL found: {best_score:.2f}")

    # Lưu model tốt nhất
    if best_model:
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', best_model)])
        joblib.dump(final_pipeline, os.path.join(save_dir, 'flight_delay_xgboost.pkl'))
        print(f"\n✅ Report chi tiết đã lưu tại: {report_path}")
    else:
        print("\n❌ Không tìm thấy model nào chạy thành công.")

if __name__ == "__main__":
    CSV_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    try:
        df = pd.read_csv(CSV_PATH)
        df = phase_2_feature_engineering(phase_1_data_cleaning(df))
        X_tr, X_te, _, _, yc_tr, yc_te, preproc = phase_3_preprocessing(df)
        train_xgboost_detailed(X_tr, X_te, yc_tr, yc_te, preproc)
    except Exception as e: 
        print(f"❌ Error Main: {e}")
        import traceback
        traceback.print_exc()