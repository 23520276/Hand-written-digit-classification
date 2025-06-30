import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os
import logging

# Đọc dữ liệu
qt = pd.read_csv("D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/qt-public.csv")
th = pd.read_csv("D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/th-public.csv")
ck = pd.read_csv("D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/ck-public.csv")

# Đổi tên cột
qt = qt.rename(columns={"diemqt": "QT"})
th = th.rename(columns={"TH": "TH"})
ck = ck.rename(columns={"CK": "CK"})

# Gộp dữ liệu
df = qt.merge(th, on="hash").merge(ck, on="hash")

# Ép kiểu dữ liệu về float
for col in ["QT", "TH", "CK"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Loại bỏ các dòng có NaN sau khi ép kiểu
df = df.dropna(subset=["QT", "TH", "CK"])

# Tính điểm trung bình tích lũy (và làm tròn đến 0.5)
df["diem_tb"] = np.round((df["QT"] * 0.2 + df["TH"] * 0.3 + df["CK"] * 0.5) * 2) / 2

# Lưu ra file
df.to_csv("D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/diem_tb.csv", index=False)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn tới thư mục dữ liệu
DUONG_DAN_DU_LIEU = "D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/"

def tai_va_tien_xu_li_du_lieu():
    try:
        logger.info("Đang đọc file annonimized.csv...")
        df = pd.read_csv(os.path.join(DUONG_DAN_DU_LIEU, "annonimized.csv"))

        cols_to_keep = ['concat(\'it001\', username)', 'is_final', 'pre_score', 'coefficient']
        df = df[cols_to_keep]

        df = df.rename(columns={
            'concat(\'it001\', username)': 'username',
            'pre_score': 'diem',
            'coefficient': 'he_so_tre'
        })

        df = df.dropna(subset=['diem', 'username'])
        df['diem'] = df['diem'] / 1000
        df['he_so_tre'] = df['he_so_tre'].fillna(100)

        logger.info("Tính toán đặc trưng tổng hợp...")
        aggregation = {
            'diem': ['mean', 'max', 'count'],
            'is_final': 'sum',
            'he_so_tre': 'mean'
        }

        dac_trung = df.groupby("username").agg(aggregation)
        dac_trung.columns = ['_'.join(col).strip() for col in dac_trung.columns.values]
        dac_trung = dac_trung.rename(columns={
            'diem_mean': 'diem_trung_binh',
            'diem_max': 'diem_cao_nhat',
            'diem_count': 'so_lan_nop',
            'is_final_sum': 'so_bai_diem_cuoi',
            'he_so_tre_mean': 'he_so_tre'
        })

        logger.info("Hoàn tất xử lý dữ liệu.")
        return dac_trung

    except Exception as e:
        logger.error(f"Lỗi trong xử lý dữ liệu: {e}")
        return None

def huan_luyen_mo_hinh(dac_trung_sinh_vien, duong_dan_diem_tb):
    try:
        logger.info("Đọc file điểm trung bình...")
        diem_tb = pd.read_csv(duong_dan_diem_tb)
        logger.info(f"Các cột trong file diem_tb.csv: {diem_tb.columns.tolist()}")

        if 'diem_tb' not in diem_tb.columns:
            logger.error("Không tìm thấy cột 'diem_tb' trong dữ liệu.")
            return None, None, None

        logger.info("Ghép đặc trưng với điểm TB...")
        du_lieu_day = pd.merge(
            dac_trung_sinh_vien.reset_index(),
            diem_tb,
            left_on="username",
            right_on="hash",
            how='inner'
        )

        y = pd.to_numeric(du_lieu_day['diem_tb'], errors='coerce')
        du_lieu_day = du_lieu_day[~y.isna()]
        y = y[~y.isna()]

        features = [
            'diem_trung_binh', 'diem_cao_nhat', 'so_lan_nop',
            'so_bai_diem_cuoi', 'he_so_tre'
        ]
        X = du_lieu_day[features]

        logger.info("Chuẩn hóa dữ liệu...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info("Chia dữ liệu train/test...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        logger.info("Huấn luyện mô hình Linear Regression...")
        mo_hinh = LinearRegression()
        mo_hinh.fit(X_train, y_train)

        y_pred = mo_hinh.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Độ chính xác R^2 (Linear Regression): {r2:.4f}")

        return mo_hinh, scaler, features

    except Exception as e:
        logger.error(f"Lỗi trong huấn luyện mô hình: {e}")
        return None, None, None

def du_doan_diem(mo_hinh, scaler, features, dac_trung_sinh_vien):
    try:
        logger.info("Dự đoán điểm trung bình...")
        X = dac_trung_sinh_vien[features]
        X_scaled = scaler.transform(X)
        diem_du_doan = mo_hinh.predict(X_scaled)
        diem_du_doan = np.round(diem_du_doan * 2) / 2  # Làm tròn 0.5

        ket_qua = pd.DataFrame({
            'MaSV': dac_trung_sinh_vien.index,
            'DiemTB_du_doan': diem_du_doan
        })

        return ket_qua

    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {e}")
        return None

def main():
    try:
        logger.info("Bắt đầu quy trình dự đoán điểm TB...")
        dac_trung = tai_va_tien_xu_li_du_lieu()
        if dac_trung is None:
            return

        duong_dan_diem_tb = os.path.join(DUONG_DAN_DU_LIEU, "diem_tb.csv")
        mo_hinh, scaler, features = huan_luyen_mo_hinh(dac_trung, duong_dan_diem_tb)

        if mo_hinh is None:
            return

        ket_qua = du_doan_diem(mo_hinh, scaler, features, dac_trung)
        if ket_qua is not None:
            duong_dan_ket_qua = os.path.join(DUONG_DAN_DU_LIEU, "ket_qua_du_doan_tb.csv")
            ket_qua.to_csv(duong_dan_ket_qua, index=False)
            logger.info(f"Đã lưu kết quả vào: {duong_dan_ket_qua}")

    except Exception as e:
        logger.error(f"Lỗi tổng: {e}")

if __name__ == "__main__":
    main()
