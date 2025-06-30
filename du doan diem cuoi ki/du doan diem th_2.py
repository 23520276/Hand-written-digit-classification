import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn tới thư mục chứa dữ liệu
DUONG_DAN_DU_LIEU = "D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/"

def tai_va_tien_xu_li_du_lieu():
    try:
        logger.info("Đang đọc file dữ liệu nộp bài...")
        df = pd.read_csv(os.path.join(DUONG_DAN_DU_LIEU, "annonimized.csv"))

        cols_to_keep = ['concat(\'it001\', username)', 'is_final', 'pre_score', 'coefficient', 'created_at', 'updated_at', 'judgement', 'status']
        df = df[cols_to_keep]

        df = df.rename(columns={
            'concat(\'it001\', username)': 'username',
            'pre_score': 'diem',
            'coefficient': 'he_so_tre'
        })
        df['diem'] = df['diem'] / 100

        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')

        df['nop_diem_0'] = (df['diem'] == 0).astype(int)
        df['nop_diem_>=_0.5'] = (df['diem'] >= 0.5).astype(int)
        df['nop_diem_>=_0.9'] = (df['diem'] >= 0.9).astype(int)
        df['ti_le_diem_cao'] = (df['diem'] >= 0.75).astype(int)
        df['thoi_gian_giai'] = (df['updated_at'] - df['created_at']).dt.total_seconds()
        df['nop_muon_24h'] = df['thoi_gian_giai'] > (24 * 3600)

        def extract_json_field(json_str, key):
            try:
                j = json.loads(json_str)
                return max(j.get(key, [0])) if isinstance(j.get(key), list) else 0
            except:
                return 0

        df['max_time'] = df['judgement'].apply(lambda x: extract_json_field(x, 'times'))
        df['max_mem'] = df['judgement'].apply(lambda x: extract_json_field(x, 'mems'))
        df['time_max_lon'] = df['max_time'] > 1000
        df['mem_max_lon'] = df['max_mem'] > 64

        df['he_so_tre'].fillna(df['he_so_tre'].mean(), inplace=True)
        df['thoi_gian_giai'].fillna(0, inplace=True)

        logger.info("Tính toán đặc trưng tổng hợp...")
        aggregation = {
            'diem': ['mean', 'max', 'min', 'std', 'median'],
            'is_final': ['sum', 'count'],
            'he_so_tre': ['mean', 'max', 'std', 'median'],
            'nop_diem_0': 'sum',
            'nop_diem_>=_0.5': 'sum',
            'nop_diem_>=_0.9': 'sum',
            'ti_le_diem_cao': 'mean',
            'thoi_gian_giai': ['mean', 'max'],
            'status': lambda x: (x == 'SCORE').sum(),
            'nop_muon_24h': 'sum',
            'mem_max_lon': 'sum',
            'time_max_lon': 'sum'
        }

        dac_trung_sinh_vien = df.groupby("username").agg(aggregation)
        dac_trung_sinh_vien.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in dac_trung_sinh_vien.columns.values]

        column_mapping = {
            'diem_mean': 'diem_trung_binh',
            'diem_max': 'diem_cao_nhat',
            'diem_min': 'diem_thap_nhat',
            'diem_std': 'diem_do_lech',
            'diem_median': 'diem_trung_vi',
            'is_final_sum': 'so_bai_diem_cuoi',
            'is_final_count': 'tong_so_bai',
            'he_so_tre_mean': 'he_so_tre_tb',
            'he_so_tre_max': 'he_so_tre_max',
            'he_so_tre_std': 'he_so_tre_do_lech',
            'he_so_tre_median': 'he_so_tre_median',
            'nop_diem_0_sum': 'so_lan_diem_0',
            'nop_diem_>=_0.5_sum': 'so_lan_diem_tren_0_5',
            'nop_diem_>=_0.9_sum': 'so_lan_diem_tren_0_9',
            'ti_le_diem_cao_mean': 'ti_le_diem_cao',
            'thoi_gian_giai_mean': 'thoi_gian_trung_binh_giai',
            'thoi_gian_giai_max': 'thoi_gian_giai_dai_nhat',
            'status_<lambda>': 'so_lan_status_SCORE',
            'nop_muon_24h_sum': 'so_lan_muon_24h',
            'mem_max_lon_sum': 'so_lan_mem_max_lon',
            'time_max_lon_sum': 'so_lan_time_max_lon'
        }

        dac_trung_sinh_vien = dac_trung_sinh_vien.rename(columns=column_mapping)
        dac_trung_sinh_vien['ti_le_bai_final'] = dac_trung_sinh_vien['so_bai_diem_cuoi'] / dac_trung_sinh_vien['tong_so_bai']
        dac_trung_sinh_vien['ti_le_diem_tren_0_5'] = dac_trung_sinh_vien['so_lan_diem_tren_0_5'] / dac_trung_sinh_vien['tong_so_bai']

        return dac_trung_sinh_vien

    except Exception as e:
        logger.error(f"Lỗi khi tải và tiền xử lý dữ liệu: {str(e)}", exc_info=True)
        return None

def huan_luyen_mo_hinh(dac_trung, path_diem):
    try:
        df_th = pd.read_csv(path_diem)
        df_th = df_th.rename(columns={'TH': 'diemqt'})
        df = pd.merge(dac_trung.reset_index(), df_th, left_on='username', right_on='hash', how='inner')

        y = pd.to_numeric(df['diemqt'], errors='coerce')
        df = df[~y.isna()]
        y = y[~y.isna()]

        X = df.drop(columns=['username', 'hash', 'diemqt'])
        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        logger.info(f"R2 score: {r2_score(y_test, y_pred):.4f}")
        return model, X.columns.tolist()

    except Exception as e:
        logger.error(f"Lỗi huấn luyện: {e}")
        return None, []

def du_doan_diem(model, features, dac_trung):
    try:
        X = dac_trung[features].fillna(0)
        y_pred = model.predict(X)
        return pd.DataFrame({
            'MaSV': dac_trung.index,
            'DiemQuaTrinh_du_doan': np.round(y_pred * 2) / 2
        })
    except Exception as e:
        logger.error(f"Lỗi dự đoán: {e}")
        return None

def main():
    dac_trung = tai_va_tien_xu_li_du_lieu()
    if dac_trung is None:
        return

    path_diem = os.path.join(DUONG_DAN_DU_LIEU, "th-public.csv")
    model, features = huan_luyen_mo_hinh(dac_trung, path_diem)
    if model is None:
        return

    kq = du_doan_diem(model, features, dac_trung)
    if kq is not None:
        output_path = os.path.join(DUONG_DAN_DU_LIEU, "ket_qua_du_doan_rf_full.csv")
        kq.to_csv(output_path, index=False)
        logger.info(f"Đã lưu kết quả vào: {output_path}")

if __name__ == "__main__":
    main()