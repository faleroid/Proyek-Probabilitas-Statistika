# Dataset dummy yang dipake buat sampel, ini bikin sendiri datanya  
X1 = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]                 # Luas rumah
X2 = [2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]                                   # Jumlah kamar
Y  = [320, 340, 365, 390, 420, 440, 465, 495, 520, 545, 570, 600, 630, 660, 700]     # Harga

# Rumus utama regresi linear ganda: Y = a + b1X1 + b2X2
# Hasil akhirnya akan merujuk ke rumus tersebut

# Perhitungan sederhana.
# Keterangan: sum adalah sigma. contoh: sum_x1y dalam math ditulis ∑X1Y

n_x1 = len(X1)
sum_x1 = sum(X1) 
sum_x2 = sum(X2)
sum_y = sum(Y)
sum_x1y = sum(x1*y for x1,y in zip(X1,Y))   #∑(X1Y) = (X1.1*Y.1) + (X1.2*Y.2) + ... + (X1.n*Y.n)
sum_x2y = sum(x2*y for x2,y in zip(X2,Y))   #∑(X2Y) = (X2.1*Y.1) + (x2.2*Y.2) + ... + (X1.n*Y.n)
sum_x1x2 = sum(x1*x2 for x1,x2 in zip(X1,X2))   #∑(X1*X2) 
sum_x1x1 = sum(x1**2 for x1 in X1)  #∑(X1)^2
sum_x2x2 = sum(x2**2 for x2 in X2)  #∑(X2)^2

"""
Didapat beberapa persamaan (merujuk dari rumus regresi linear berganda):
∑Y = n(X1) + ∑X1 + ∑X2 ...(1)
∑X1Y = ∑X1 + ∑X1² + ∑X1X2 ...(2)
∑X2Y = ∑X2 + ∑X1X2 + ∑X2² ...(3)
"""

# Buat matriks 3*3 dari ketiga persamaan di atas
A = [
    [n_x1, sum_x1, sum_x2],
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

print(f"Persamaan regresi: Y = {b0:.2f} + {b1:.2f}*X1 + {b2:.2f}*X2")   # Y = 101.49 + 4.75*X1 + 10.81*X2

# Intinya cuma ada dua input dengan satu output
x1_pred = float(input("masukkan luas tanah: "))
x2_pred = int(input("masukkan jumlah kamar: "))
y_pred = b0 + b1 * x1_pred + b2 * x2_pred
print(f"Prediksi harga rumah {x1_pred} m^2, {x2_pred} kamar: {y_pred:.2f} juta")