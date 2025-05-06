import pandas as pd

from flask import Flask, render_template, request
app = Flask(__name__)
# Fungsi sum manual
def manual_sum(data):
    total = 0
    for val in data:
        total += val
    return total

# Fungsi sigma hasil perkalian 2 list (contoh: X1*Y)
def manual_sum_product(list1, list2):
    total = 0
    for a, b in zip(list1, list2):
        total += a * b
    return total

# Fungsi sigma kuadrat dari list (contoh: X1^2)
def manual_sum_square(data):
    total = 0
    for val in data:
        total += val * val
    return total

df_train = pd.read_csv('student_habits_performance.csv')
df_clean = df_train[df_train['gender'].str.lower() != 'other']

X1 = df_clean['study_hours_per_day']
X2 = df_clean['social_media_hours']
Y = df_clean['exam_score']


# for feature in df_train.columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=df_train[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()


# Rumus utama regresi linear ganda: Y = b0 + b1X1 + b2X2
# Hasil akhirnya akan merujuk ke rumus tersebut

# Perhitungan sederhana.
# Keterangan: sum adalah sigma. contoh: sum_x1y dalam math ditulis ∑X1Y

n = len(X1)
sum_x1 = manual_sum(X1)
sum_x2 = manual_sum(X2)
sum_y = manual_sum(Y)
sum_x1y = manual_sum_product(X1, Y)
sum_x2y = manual_sum_product(X2, Y)
sum_x1x2 = manual_sum_product(X1, X2)
sum_x1x1 = manual_sum_square(X1)
sum_x2x2 = manual_sum_square(X2)

"""
Didapat beberapa persamaan (merujuk dari rumus regresi linear berganda):
∑Y = n(X1) + ∑X1 + ∑X2 ...(1)
∑X1Y = ∑X1 + ∑X1² + ∑X1X2 ...(2)
∑X2Y = ∑X2 + ∑X1X2 + ∑X2² ...(3)
"""

# Buat matriks 3*3 dari ketiga persamaan di atas
A = [
    [n, sum_x1, sum_x2],
    [sum_x1, sum_x1x1, sum_x1x2],
    [sum_x2, sum_x1x2, sum_x2x2]
]
C = [sum_y, sum_x1y, sum_x2y]

# Mencari koefisien dengan menggunakan gauss_jordan method (paling masuk akal buat di program non-library)
def gauss_elimination(A, C):
    n = len(C)
    for i in range(n):
        factor = A[i][i]    # Awalnya: Baris 0 kolom 0 (Satu utama)
        for j in range(i, n):   #setiap baris ke-i kolom ke-j dibagi faktor
            A[i][j] /= factor
        C[i] /= factor
        
        # Hilangkan elemen di bawahnya
        for k in range(i+1, n):
            factor = A[k][i]    # Awalnya: Baris 1 kolom 0 (Satu utama)
            for j in range(i, n):
                A[k][j] -= factor * A[i][j] 
            C[k] -= factor * C[i]

    # Ini tinggal finishing, pindah ruas untuk dapet hasilnya
    x = [0,0,0]
    for i in range(n-1, -1, -1):
        x[i] = C[i] - sum(A[i][j]*x[j] for j in range(i+1, n))
    return x

b0, b1, b2 = gauss_elimination(A, C)

y_preds = [b0 + b1 * x1 + b2 * x2 for x1, x2 in zip(X1, X2)]
# Hitung MAE, MSE, RMSE manual
n = len(Y)
sum_abs_error = 0
sum_squared_error = 0

for y_actual, y_pred in zip(Y, y_preds):
    error = y_actual - y_pred
    sum_abs_error += abs(error)
    sum_squared_error += error ** 2

mae = sum_abs_error / n
mse = sum_squared_error / n
rmse = mse ** 0.5
mean_y = manual_sum(Y) / n
ss_tot = manual_sum((y - mean_y) ** 2 for y in Y)
r_squared = 1 - (sum_squared_error / ss_tot)

# Tampilkan hasil
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R² = {r_squared:.4f}")

print(f"Persamaan regresi: Y = {b0:.2f} + {b1:.2f}*X1 + {b2:.2f}*X2")

# x1_pred = float(input("Lama Belajar(jam): "))
# x2_pred = int(input("Social Media(jam): "))
# y_pred = b0 + b1 * x1_pred + b2 * x2_pred

# if y_pred < 0:
#     print("Tidak masuk akal, mending lu stop main sosmed")
# else:
#     print(f'Prediksi Nilai Ujian: {y_pred:.2f}')


@app.route('/', methods=['GET', 'POST'])
def index():
    y_pred = None
    if request.method == 'POST':
        try:
            study_hours = float(request.form['x1'])
            social_media = float(request.form['x2'])
            y_pred = b0 + b1 * study_hours + b2 * social_media
        except (KeyError, ValueError):
            y_pred = "Input tidak valid"

    return render_template('index.html', b0=b0, b1=b1, b2=b2, y_pred=y_pred) 

if __name__ == '__main__':
    app.run(debug=True)