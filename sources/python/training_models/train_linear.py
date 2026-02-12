import pandas as pd
import numpy as np
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ==============================================================================
# PHASE 1: DATA CLEANING
# ==============================================================================
def phase_1_data_cleaning(df):
    print("🔹 PHASE 1: DATA CLEANING...")
    cols = ['ARR_DELAY', 'DEP_DELAY', 'CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 
            'OP_CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK', 
            'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
    
    # Chỉ giữ các cột tồn tại trong dataframe
    cols_to_keep = [c for c in cols if c in df.columns]
    df_clean = df[cols_to_keep].copy()
    
    # Fill NA thời tiết
    df_clean[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']] = df_clean[['O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']].fillna(0)
    
    # Drop dòng thiếu Delay
    return df_clean.dropna(subset=['ARR_DELAY', 'DEP_DELAY'])

# ==============================================================================
# PHASE 2: FEATURE ENGINEERING
# ==============================================================================
def phase_2_feature_engineering(df):
    print("🔹 PHASE 2: FEATURE ENGINEERING...")
    
    # 1. Target
    df['ARR_DELAY'] = pd.to_numeric(df['ARR_DELAY'], errors='coerce').fillna(0)
    df['DEP_DELAY'] = pd.to_numeric(df['DEP_DELAY'], errors='coerce').fillna(0)
    df['TOTAL_DELAY'] = df['ARR_DELAY'] + df['DEP_DELAY']
    
    df['DELAY_LEVEL'] = df['TOTAL_DELAY'].apply(lambda x: 0 if x<=0 else (1 if x<=45 else (2 if x<=90 else 3)))
    
    # 2. DateTime Parsing (Giờ bay)
    try:
        # Chuyển sang datetime object
        df['CRS_DEP_TIME_DT'] = pd.to_datetime(df['CRS_DEP_TIME'], errors='coerce')
        df['DEP_HOUR'] = df['CRS_DEP_TIME_DT'].dt.hour
        df['DEP_HOUR'] = df['DEP_HOUR'].fillna(12).astype(int)
    except:
        df['DEP_HOUR'] = 12

    # 3. Features khác
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)
    
    def get_season(m):
        try: return {3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall'}.get(int(m), 'Winter')
        except: return 'Unknown'
    df['SEASON'] = df['MONTH'].apply(get_season)
    
    # 4. Route Risk
    route_stats = df.groupby(['ORIGIN', 'DEST'])['TOTAL_DELAY'].mean().reset_index().rename(columns={'TOTAL_DELAY': 'ROUTE_RISK'})
    df = df.merge(route_stats, on=['ORIGIN', 'DEST'], how='left')
    df['ROUTE_RISK'] = df['ROUTE_RISK'].fillna(df['TOTAL_DELAY'].mean())
    
    # 5. Grouping Origin/Dest
    df['ORIGIN'] = df['ORIGIN'].astype(str)
    df['DEST'] = df['DEST'].astype(str)
    top_20_org = df['ORIGIN'].value_counts().nlargest(20).index
    df['ORIGIN_GROUPED'] = df['ORIGIN'].apply(lambda x: x if x in top_20_org else 'OTHER')
    
    top_20_dst = df['DEST'].value_counts().nlargest(20).index
    df['DEST_GROUPED'] = df['DEST'].apply(lambda x: x if x in top_20_dst else 'OTHER')
    
    # Drop cột thừa
    cols_drop = ['ARR_DELAY', 'DEP_DELAY', 'CRS_DEP_TIME', 'CRS_DEP_TIME_DT', 'ORIGIN', 'DEST']
    return df.drop(columns=[c for c in cols_drop if c in df.columns], errors='ignore')

# ==============================================================================
# PHASE 3: PREPROCESSING (ĐÃ FIX LỖI)
# ==============================================================================
def phase_3_preprocessing(df):
    print("🔹 PHASE 3: PREPROCESSING...")
    
    X = df.drop(columns=['TOTAL_DELAY', 'DELAY_LEVEL'])
    y_reg = df['TOTAL_DELAY']
    y_clf = df['DELAY_LEVEL']
    
    num_cols = ['CRS_ELAPSED_TIME', 'ROUTE_RISK', 'O_PRCP', 'O_WSPD', 'D_PRCP', 'D_WSPD']
    cat_cols = ['OP_CARRIER', 'ORIGIN_GROUPED', 'DEST_GROUPED', 'DEP_HOUR', 'MONTH', 'SEASON', 'IS_WEEKEND']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    # FIX LỖI Ở ĐÂY: Tách biến ra rồi return tuple rõ ràng
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor

# ==============================================================================
# PHASE 4: TUNING & TRAINING FOR LINEAR REGRESSION
# ==============================================================================
def train_linear(X_train, X_test, y_train, y_test, y_clf_test, preprocessor):
    print("\n🚀 TRAINING LINEAR REGRESSION...")
    save_dir, report_dir = 'models', 'data/report'
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    # === TRAIN ===
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression(fit_intercept=True, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # === EVALUATE ===
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Convert to Class for comparison
    def to_class(m):
        if m <= 0: return 0
        elif m <= 45: return 1
        elif m <= 90: return 2
        else: return 3
        
    y_pred_class = [to_class(m) for m in y_pred]
    acc = accuracy_score(y_clf_test, y_pred_class)
    
    print(f"   -> RMSE: {rmse:.2f} | R2: {r2:.4f} | Conv. Accuracy: {acc:.2%}")
    
    # === REPORT ===
    report_path = os.path.join(report_dir, 'linear_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== LINEAR REGRESSION REPORT ===\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"Accuracy (Converted): {acc:.4f}\n\n")
        f.write("Detailed Classification Report (Converted):\n")
        f.write(classification_report(y_clf_test, y_pred_class, target_names=['Good', 'Minor', 'Moderate', 'Severe']))
        
    # === SAVE ===
    model_path = os.path.join(save_dir, 'flight_delay_linear.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Report saved at: {report_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    CSV_PATH = "D:/UEL/ADA/ADA-Flight-Delay/data/sampled/flight_data_sampled_2023.csv"
    
    try:
        df = pd.read_csv(CSV_PATH)
        df_p1 = phase_1_data_cleaning(df)
        df_p2 = phase_2_feature_engineering(df_p1)
        
        # Unpack đúng 7 giá trị trả về
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, preproc = phase_3_preprocessing(df_p2)
        
        # Gọi hàm train
        train_linear(X_tr, X_te, yr_tr, yr_te, yc_te, preproc)
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {CSV_PATH}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()