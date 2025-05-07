import pandas as pd
from math import floor

from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)


# Fungsi manual
def manual_sum(data):
    total = 0
    for val in data:
        total += val
    return total

def manual_sum_product(list1, list2):
    total = 0
    for a, b in zip(list1, list2):
        total += a * b
    return total

def manual_sum_square(data):
    total = 0
    for val in data:
        total += val * val
    return total

# Load dan filter data
df = pd.read_csv('student_habits_performance.csv')
df_clean = df[df['gender'].str.lower() != 'other']
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle manual

# Pisah fitur
X_all = df_clean['study_hours_per_day'].tolist()
Y_all = df_clean['exam_score'].tolist()

# Split 80:20
n_total = len(X_all)
n_train = floor(0.8 * n_total)

X_train = X_all[:n_train]
Y_train = Y_all[:n_train]

X_test = X_all[n_train:]
Y_test = Y_all[n_train:]

# TRAINING - hitung koefisien regresi
sum_x = manual_sum(X_train)
sum_y = manual_sum(Y_train)
sum_xy = manual_sum_product(X_train, Y_train)
sum_xx = manual_sum_square(X_train)
mean_x = sum_x / n_train
mean_y = sum_y / n_train

b1 = (n_train * sum_xy - sum_x * sum_y) / (n_train * sum_xx - sum_x ** 2)
b0 = mean_y - b1 * mean_x

print(f"\nMODEL LATIHAN:\nPersamaan regresi: Y = {b0:.2f} + {b1:.2f}*X")

# Uji t untuk b1
residuals = [(y - (b0 + b1 * x)) for x, y in zip(X_train, Y_train)]
s_squared = sum(r**2 for r in residuals) / (n_train - 2)
s = s_squared ** 0.5
std_error_b1 = s / ((manual_sum_square(X_train) - n_train * mean_x ** 2) ** 0.5)
t_hit = b1 / std_error_b1

print(f"Uji t untuk b1: t hitung = {t_hit:.4f} (bandingkan dengan t tabel df = {n_train - 2})")

# TESTING - Evaluasi prediksi
y_preds = [b0 + b1 * x for x in X_test]
n_test = len(Y_test)

sum_abs_error = 0
sum_squared_error = 0
mean_y_test = manual_sum(Y_test) / n_test

for y, y_pred in zip(Y_test, y_preds):
    err = y - y_pred
    sum_abs_error += abs(err)
    sum_squared_error += err ** 2

mae = sum_abs_error / n_test
mse = sum_squared_error / n_test
rmse = mse ** 0.5
ss_tot = sum((y - mean_y_test) ** 2 for y in Y_test)
r_squared = 1 - (sum_squared_error / ss_tot)

# print("\nEVALUASI DI DATA TEST:")
# print(f"MAE  = {mae:.2f}")
# print(f"MSE  = {mse:.2f}")
# print(f"RMSE = {rmse:.2f}")
# print(f"R²   = {r_squared:.4f}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            study_hours = int(request.form['study_hours'])
            user_name = request.form['name']
            prediction = b0 + b1 * study_hours
            prediction = round(prediction, 2)
            # Redirect ke halaman hasil dan kirim hasil prediksi lewat URL
            return redirect(url_for('hasil', user_name=user_name, prediction=prediction))
        except:
            return redirect(url_for('hasil', prediction="Input tidak valid"))
    return render_template('index.html')

@app.route('/hasil')
def hasil():
    prediction = request.args.get('prediction')  # Ambil dari URL query
    user_name = request.args.get('user_name')  # Ambil dari URL query
    return render_template('output.html', name=user_name, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

