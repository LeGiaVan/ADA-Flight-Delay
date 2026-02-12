import pandas as pd
import numpy as np
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier # <--- THÊM THƯ VIỆN NÀY
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
def train_logistic_detailed(X_train, X_test, y_train, y_test, preprocessor):
    print("\n🚀 TRAINING LOGISTIC REGRESSION (DETAILED PER-CLASS REPORT)...")
    save_dir, report_dir = 'models', 'data/report'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    # Định nghĩa Grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }
    
    print("   ⚙️ Preprocessing data...")
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    
    report_path = os.path.join(report_dir, 'logistic_detailed_report.txt')
    best_score = -1
    best_model = None
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== LOGISTIC REGRESSION: PER-CLASS PERFORMANCE ===\n\n")
        header = (f"{'C':<6} | {'Solver':<10} || "
                  f"{'GOOD (0)':<12} | {'MINOR (1)':<12} | {'MODERATE (2)':<12} | {'SEVERE (3)':<12}\n")
        sub_header = (f"{'':<19}|| "
                      f"{'P':<5} {'R':<5} | {'P':<5} {'R':<5} | {'P':<5} {'R':<5} | {'P':<5} {'R':<5}\n")
        
        f.write(header)
        f.write(sub_header)
        f.write("-" * 110 + "\n")
        
        print(f"   ⚙️ Đang chạy từng tham số và tính toán...")
        
        for params in ParameterGrid(param_grid):
            # In ra console để theo dõi
            print(f"   -> Testing C={params['C']}, solver={params['solver']}...", end="")
            
            try:
                # === CHIẾN THUẬT SỬA LỖI ===
                # Nếu solver là liblinear, ta phải dùng OneVsRestClassifier để bọc nó lại
                if params['solver'] == 'liblinear':
                    base_clf = LogisticRegression(
                        C=params['C'], 
                        solver='liblinear',
                        class_weight='balanced', 
                        max_iter=5000, 
                        random_state=42
                    )
                    clf = OneVsRestClassifier(base_clf) # Bọc lại
                else:
                    # Nếu là lbfgs thì chạy bình thường
                    clf = LogisticRegression(
                        C=params['C'], 
                        solver='lbfgs',
                        class_weight='balanced', 
                        max_iter=5000, 
                        random_state=42
                    )

                clf.fit(X_train_proc, y_train)
                y_pred = clf.predict(X_test_proc)
                
                # Lấy report dạng dict
                report = classification_report(y_test, y_pred, labels=[0,1,2,3], output_dict=True)
                
                p0, r0 = report['0']['precision'], report['0']['recall']
                p1, r1 = report['1']['precision'], report['1']['recall']
                p2, r2 = report['2']['precision'], report['2']['recall']
                p3, r3 = report['3']['precision'], report['3']['recall']
                
                row = (f"{params['C']:<6} | {params['solver']:<10} || "
                       f"{p0:.2f}  {r0:.2f}  | {p1:.2f}  {r1:.2f}  | {p2:.2f}  {r2:.2f}  | {p3:.2f}  {r3:.2f}\n")
                f.write(row)
                print(" OK")
                
                if r3 > best_score:
                    best_score = r3
                    best_model = clf
                    
            except Exception as e:
                print(f" LỖI: {e}")
                f.write(f"{params['C']:<6} | {params['solver']:<10} || ERROR: {str(e)}\n")

        f.write("-" * 110 + "\n")
        if best_model:
            f.write(f"Best Model Selected based on SEVERE RECALL: {best_score:.2f}")
        else:
            f.write("No successful model found.")

    if best_model is not None:
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', best_model)])
        joblib.dump(final_pipeline, os.path.join(save_dir, 'flight_delay_logistic.pkl'))
        print(f"\n✅ Report chi tiết đã lưu tại: {report_path}")
    else:
        print("\n❌ LỖI: Không tìm thấy model nào chạy thành công.")

if __name__ == "__main__":
    CSV_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    try:
        df = pd.read_csv(CSV_PATH)
        df = phase_2_feature_engineering(phase_1_data_cleaning(df))
        X_tr, X_te, _, _, yc_tr, yc_te, preproc = phase_3_preprocessing(df)
        train_logistic_detailed(X_tr, X_te, yc_tr, yc_te, preproc)
    except Exception as e: 
        print(f"❌ Error Main: {e}")