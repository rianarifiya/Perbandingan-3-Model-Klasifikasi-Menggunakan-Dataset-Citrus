# 🍊 UTS – Klasifikasi Buah: Orange vs Grapefruit

> **Ujian Tengah Semester – Machine Learning**  
> Perbandingan tiga algoritma klasifikasi menggunakan dataset Citrus dari Kaggle

---

## 📂 Struktur Proyek

```
uts_klasifikasi/
├── citrus.csv                          # Dataset dari Kaggle
├── README.md                           # Dokumentasi ini
│
├── klasifikasi_naive_bayes.py          # Model 1: Naive Bayes
├── klasifikasi_decision_tree.py        # Model 2: Decision Tree
├── klasifikasi_svm.py                  # Model 3: Support Vector Machine
│
├── [Output Naive Bayes]
│   ├── nb_dist_diameter.png
│   ├── nb_dist_weight.png
│   ├── nb_heatmap_korelasi.png
│   ├── nb_scatter_diameter_weight.png
│   ├── nb_confusion_matrix.png
│   ├── nb_precision_recall_curve.png
│   └── nb_roc_curve.png
│
├── [Output Decision Tree]
│   ├── dt_dist_diameter.png
│   ├── dt_dist_weight.png
│   ├── dt_heatmap_korelasi.png
│   ├── dt_scatter_diameter_weight.png
│   ├── dt_visualisasi_pohon.png
│   ├── dt_feature_importance.png
│   ├── dt_confusion_matrix.png
│   ├── dt_precision_recall_curve.png
│   └── dt_roc_curve.png
│
└── [Output SVM]
    ├── svm_dist_diameter.png
    ├── svm_dist_weight.png
    ├── svm_heatmap_korelasi.png
    ├── svm_scatter_diameter_weight.png
    ├── svm_confusion_matrix.png
    ├── svm_precision_recall_curve.png
    └── svm_roc_curve.png
```

---

## 📦 Dataset

**Sumber:** [Kaggle – Oranges vs Grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)

### Informasi Dataset

| Fitur      | Tipe    | Keterangan                          |
|-----------|---------|-------------------------------------|
| `name`     | string  | Label kelas: `orange` / `grapefruit`|
| `diameter` | float   | Diameter buah (cm)                  |
| `weight`   | float   | Berat buah (gram)                   |
| `red`      | int     | Nilai warna merah (0–255)           |
| `green`    | int     | Nilai warna hijau (0–255)           |
| `blue`     | int     | Nilai warna biru (0–255)            |

- **Jumlah data:** 10.000 baris
- **Jumlah kelas:** 2 (orange = 5.000, grapefruit = 5.000)
- **Missing value:** Tidak ada

---

## 🔄 Tahapan Pembuatan Model (Berlaku untuk Ketiga Model)

### 1. Pengumpulan dan Persiapan Data

#### Load Library
Setiap file dimulai dengan mengimpor semua library yang dibutuhkan:
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn import metrics
```

#### Load & Eksplorasi Data
- Membaca dataset dengan `pd.read_csv('citrus.csv')`
- Menampilkan statistik deskriptif (`describe()`) untuk melihat mean, std, min, max setiap fitur
- Menampilkan jumlah data per kelas (`value_counts()`)

#### Distribusi Data
Menampilkan distribusi dua fitur utama menggunakan `sns.distplot()`:
- Distribusi `diameter` → memperlihatkan sebaran ukuran buah
- Distribusi `weight` → memperlihatkan sebaran berat buah

#### Label Encoding
Kolom `name` yang bertipe string diubah menjadi angka menggunakan `LabelEncoder`:
```python
le = LabelEncoder()
df_citrus['name'] = le.fit_transform(df_citrus['name'])
# grapefruit = 0, orange = 1
```

#### Correlation Matrix & Heatmap
Menghitung korelasi antar fitur untuk memahami hubungan antar variabel:
```python
df_citrus.corr()
sns.heatmap(df_citrus.corr(), annot=True, fmt='.2f', cmap='coolwarm')
```
Dari hasil korelasi terlihat bahwa `diameter` dan `weight` memiliki korelasi yang sangat kuat (-0.77) terhadap label kelas `name`.

---

### 2. Pembagian Data (Training dan Testing)

#### Scatter Plot
Sebelum split, ditampilkan scatter plot hubungan antara `diameter` dan `weight` untuk melihat sebaran data per kelas secara visual.

#### Definisi Fitur dan Target
```python
X = df_citrus[['diameter', 'weight', 'red', 'green', 'blue']].values  # fitur
y = df_citrus['name'].values   # target: 0=grapefruit, 1=orange
```

#### Train-Test Split (75:25)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# Data train: 7.500 | Data test: 2.500
```

#### Feature Scaling
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
```
> **Catatan:** Scaling penting untuk Naive Bayes dan SVM agar semua fitur berada pada skala yang sama. Untuk Decision Tree, scaling tidak wajib namun tetap dilakukan untuk konsistensi.

---

### 3. Membuat Model Klasifikasi

#### 🌳 Model 1 — Decision Tree (`klasifikasi_decision_tree.py`)

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
classifier.fit(X_train, y_train)
```

**Cara kerja:** Decision Tree membangun struktur pohon dengan membagi data secara rekursif berdasarkan fitur yang memberikan Gini Impurity terkecil. Setiap node merepresentasikan kondisi (mis. `diameter ≤ 0.5`), dan setiap daun merepresentasikan prediksi kelas.

**Parameter:**
- `max_depth=5` — membatasi kedalaman pohon untuk mencegah overfitting
- `criterion='gini'` — menggunakan Gini Impurity sebagai ukuran pemisahan

File ini juga menghasilkan visualisasi tambahan:
- **Visualisasi pohon** (`plot_tree`) untuk melihat struktur keputusan
- **Feature Importance** untuk melihat fitur mana yang paling berpengaruh

---

#### 📊 Model 2 — Naive Bayes (`klasifikasi_naive_bayes.py`)

```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

**Cara kerja:** Naive Bayes menggunakan Teorema Bayes untuk menghitung probabilitas posterior setiap kelas. Disebut "naive" karena mengasumsikan setiap fitur bersifat independen. Versi Gaussian mengasumsikan distribusi fitur mengikuti distribusi normal.

**Parameter:** Tidak ada hyperparameter utama — prior dihitung langsung dari data training.

---

#### 🤖 Model 3 — SVM (`klasifikasi_svm.py`)

```python
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
classifier.fit(X_train, y_train)
```

**Cara kerja:** SVM mencari hyperplane yang memaksimalkan margin antara dua kelas. Kernel RBF (Radial Basis Function) memungkinkan SVM memisahkan data yang tidak dapat dipisahkan secara linear dengan memetakan data ke dimensi yang lebih tinggi.

**Parameter:**
- `kernel='rbf'` — menggunakan kernel non-linear
- `C=1.0` — parameter regularisasi
- `gamma='scale'` — skala kernel otomatis berdasarkan jumlah fitur
- `probability=True` — mengaktifkan estimasi probabilitas untuk kurva ROC dan PR

---

### 4. Membuat Report Hasil Klasifikasi

Setiap model menghasilkan laporan yang sama:

```python
# Prediksi
y_pred = classifier.predict(X_test)

# Accuracy
accuracy_score(y_test, y_pred)

# Classification Report (Precision, Recall, F1 per kelas)
print(classification_report(y_test, y_pred, target_names=['grapefruit', 'orange']))

# F1 Score keseluruhan
print(f"F1 Score : {f1_score(y_test, y_pred)}")
```

---

### 5. Melakukan Evaluasi

#### Confusion Matrix
```python
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
```
Menampilkan jumlah TP, TN, FP, FN dalam bentuk heatmap untuk melihat di mana model membuat kesalahan prediksi.

#### Precision-Recall Curve
```python
y_pred_proba = classifier.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
ax.plot(recall, precision)
```
Memperlihatkan trade-off antara Precision dan Recall pada berbagai threshold keputusan.

#### ROC Curve (AUC)
```python
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, color='firebrick')
```
Memperlihatkan kemampuan model dalam membedakan dua kelas. Semakin mendekati sudut kiri atas, semakin baik model.

---

## 📊 Hasil Evaluasi

### Perbandingan Metrik (Test Set 25%)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Decision Tree** | 92.48% | 0.93 | 0.92 | 0.92 |
| **Naive Bayes** | 92.08% | 0.92 | 0.92 | 0.92 |
| **SVM** ⭐ | **93.68%** | **0.94** | **0.94** | **0.94** |

---

## 🏆 Kesimpulan

**1. Decision Tree (Accuracy: 92.48%)**
Performa baik dan mudah diinterpretasi. Dari feature importance, `diameter` dan `weight` terbukti menjadi fitur yang paling dominan dalam membedakan orange dari grapefruit. Visualisasi pohon keputusan memudahkan pemahaman logika klasifikasi.

**2. Naive Bayes (Accuracy: 92.08%)**
Meskipun memiliki asumsi independensi antar fitur (yang tidak sepenuhnya terpenuhi karena `diameter` dan `weight` berkorelasi sangat tinggi), Naive Bayes tetap memberikan performa yang kompetitif dengan waktu training paling cepat di antara ketiga model.

**3. SVM (Accuracy: 93.68%) ✅ TERBAIK**
SVM dengan kernel RBF menghasilkan akurasi tertinggi. Kernel RBF mampu menangkap batas keputusan non-linear antara dua kelas, sehingga lebih fleksibel dibandingkan model lainnya. Konsisten unggul di semua metrik (Precision, Recall, F1).

> **Rekomendasi:** Gunakan **SVM** jika prioritas utama adalah akurasi. Gunakan **Decision Tree** jika interpretabilitas model lebih penting.

---

## ▶️ Cara Menjalankan

```bash
# 1. Install dependensi
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Pastikan citrus.csv berada di folder yang sama

# 3. Jalankan masing-masing model
python klasifikasi_naive_bayes.py
python klasifikasi_decision_tree.py
python klasifikasi_svm.py
```

---

## 🔧 Dependensi

| Library | Fungsi |
|---------|--------|
| `pandas` | Manipulasi data tabular |
| `numpy` | Operasi numerik |
| `scikit-learn` | Model ML, preprocessing, evaluasi |
| `matplotlib` | Visualisasi grafik |
| `seaborn` | Visualisasi statistik |

---

*UTS Machine Learning – Klasifikasi Buah Citrus (Orange vs Grapefruit)*
