import pandas as pd

df = pd.read_csv("D:/code_things/do an cuoi ki mon may hoc/du doan diem cuoi ki/wecode.data/annonimized.csv")
so_luong_sinh_vien = df['concat(\'it001\', username)'].nunique()
print("Số lượng sinh viên dự đoán được:", so_luong_sinh_vien)