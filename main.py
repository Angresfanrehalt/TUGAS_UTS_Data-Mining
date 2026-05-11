import pandas as pd
import numpy as np

# 1. Membuat DataFrame dengan data yang lebih kompleks (ada nilai NaN/Kosong)
data = {
    'Studentid': ['S1', 'S2', 'S3', 'S4', 'S5'],
    'Age': [21, 19, 22, 20, np.nan],  # Ada data hilang
    'Mark': [85, 90, 78, 92, 40],     # 30%
    'UTS': [85, 90, 78, 92, 40],      # 30%
    'UAS': [85, 90, 78, 70, 80],      # 50%
    'Present': [10, 14, 14, 13, 10],  # max 14 (20%)
    'Class': ['A', 'B', 'A', 'B', 'A']
}

df = pd.DataFrame(data)

print("---- DataFrame Sebelum Cleaning ----")
print(df)

# 2. Data Cleaning: Mengisi nilai Age yang kosong dengan rata-rata
df['Age'] = df['Age'].fillna(df['Age'].mean())

def grade_count(x):
    if x >= 80:
        return "A"
    elif x < 50:
        return "E"
    elif 50 <= x < 65:
        return "D"
    elif 65 <= x < 75:
        return "C"
    else:
        return "B"

def calculate_total(present, uts, uas):
    present = present / 14 * 20  # 20%
    uts = uts / 100 * 30         # 30%
    uas = uas / 100 * 50         # 50%
    return present + uts + uas

# 3. Feature Engineering: Menambah kolom status berdasarkan nilai
df['Grade'] = df['Mark'].apply(lambda x: grade_count(x))
df['Status'] = np.where(df['Mark'] >= 75, 'Lulus', 'Tidak Lulus')
df['student_number'] = df['Studentid'].apply(lambda x: x.replace('S', '001-0'))

#. implement calculate_total() into df['total'] column
df['total'] = df.apply(lambda row: calculate_total(row['Present'], row['UTS'], row['UAS']), axis=1)

df = df.drop(columns=['Studentid'])

# 4. Aggregation: Menghitung rata-rata nilai per kelas
class_summary = df.groupby('Class')['Mark'].mean()

print("---- DataFrame Setelah Cleaning ----")
print(df)

print("\n---- Rata-rata Nilai per Kelas ----")
print(class_summary)

# 5. Statistik Deskriptif
print("\n---- Ringkasan Statistik ----")
print(df.describe())