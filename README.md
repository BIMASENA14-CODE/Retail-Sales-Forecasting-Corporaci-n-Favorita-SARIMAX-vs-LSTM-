# Retail Sales Forecasting — Corporación Favorita (SARIMAX vs LSTM)

Proyek ini membandingkan dua pendekatan forecasting time series — SARIMAX (model statistik) dan LSTM (deep learning) — untuk memprediksi penjualan harian pada lima kategori produk dari dataset retail Corporación Favorita (Ecuador), serta menyediakan dashboard interaktif untuk eksplorasi hasilnya.

## Tim
- Bima Sena (24031554214)
- Fawazul Ammar (24031554136)
- Satya Apta Mahardika (24031554024)

Pembimbing: Ulfa Siti Nuraini, S.Stat., M.Stat.
Program Studi S1 Sains Data, FMIPA, Universitas Negeri Surabaya.

## Tentang Dataset

Dataset bersumber dari kompetisi Kaggle *Store Sales – Time Series Forecasting*, berisi data penjualan harian 2013–2017 pada 54 toko dengan 33 kategori produk. Lima file (train, oil price, stores, transactions, holidays/events) digabungkan menjadi satu dataset terintegrasi.

Lima kategori dengan kontribusi volume penjualan tertinggi dipilih sebagai fokus analisis: **Grocery I, Beverages, Produce, Cleaning, dan Dairy**.

## Metodologi

1. **Integrasi & pembersihan data** — left join lima sumber data berdasarkan `date` dan `store_nbr`, penanganan missing values (interpolasi linier untuk harga minyak, imputasi nol untuk transaksi).
2. **Rekayasa fitur** — fitur temporal (tahun, bulan, minggu, hari dalam seminggu) dan fitur kontekstual biner (hari libur, tanggal gajian, akhir pekan).
3. **Pembagian data** — split temporal 80% data latih / 20% data uji (bukan random split, untuk menjaga urutan waktu).
4. **Pemodelan SARIMAX** — dilakukan uji stasioneritas (ADF test), analisis ACF/PACF, lalu fitting SARIMAX(1,1,1)(1,1,0,7) per kategori dengan variabel eksogen (harga minyak, promosi, hari libur, tanggal gajian).
5. **Pemodelan LSTM** — windowing data multivariate, arsitektur 2-layer LSTM (64 dan 32 unit) dengan Dropout, dilatih dengan EarlyStopping dan ReduceLROnPlateau.
6. **Evaluasi** — RMSE, MAE, dan MAPE dihitung pada data uji untuk kedua model, dibandingkan per kategori.

## Hasil

Performa kedua model **bervariasi tergantung kategori produk** — tidak ada satu model yang unggul mutlak di semua kategori:

| Kategori   | SARIMAX MAPE | LSTM MAPE | Model Lebih Baik |
|------------|:---:|:---:|:---:|
| Grocery I  | 33.9% | 57.0% | SARIMAX |
| Beverages  | 35.4% | 32.2% | LSTM (tipis) |
| Produce    | 80.6% | 44.6% | LSTM |
| Cleaning   | 47.3% | 61.3% | SARIMAX |
| Dairy      | 37.4% | 43.5% | SARIMAX |

**Temuan utama:**
- SARIMAX cenderung lebih kompetitif pada kategori dengan pola musiman yang stabil dan terstruktur (Dairy, Cleaning), dan koefisien variabel eksogennya mudah diinterpretasikan untuk kebutuhan bisnis.
- LSTM lebih unggul pada kategori dengan volatilitas tinggi seperti Produce, karena mampu menangkap pola non-linear tanpa asumsi stasioneritas.
- Variabel hari libur dan tanggal gajian terbukti relevan pada kedua pendekatan, mengonfirmasi pengaruh kalender dan siklus pendapatan terhadap perilaku belanja.

**Keterbatasan:** model dilatih per kategori secara agregat (bukan per toko), sehingga akurasi berpotensi ditingkatkan dengan pemodelan granular per toko atau pendekatan hybrid.

## Struktur Repository

```
.
├── notebook/
│   └── Data_Mining_Kelompok_09.ipynb     # Notebook utama: EDA, preprocessing, SARIMAX, LSTM
├── dashboard/
│   └── dashboard.py                       # Dashboard interaktif Streamlit
├── artifacts/                             # Model & hasil prediksi tersimpan (dibuat dari notebook)
│   ├── sarimax_models/
│   ├── lstm_models/
│   ├── predictions.parquet
│   ├── metrics_comparison.csv
│   ├── rekomendasi_model.csv
│   ├── historical_sales.parquet
│   └── metadata.json
└── README.md
```

## Cara Menjalankan

### 1. Jalankan notebook

Buka `notebook/Data_Mining_Kelompok_09.ipynb` dan jalankan seluruh cell secara berurutan, termasuk cell terakhir "Simpan Artifacts" yang akan membuat folder `artifacts/` berisi model dan hasil prediksi.

### 2. Jalankan dashboard

```bash
pip install streamlit pandas numpy plotly statsmodels tensorflow pyarrow
streamlit run dashboard/dashboard.py
```

Pastikan folder `artifacts/` berada di lokasi yang sama (atau sesuaikan path `ARTIFACT_DIR` di `dashboard.py`) dengan file `dashboard.py`.

Dashboard akan terbuka otomatis di browser pada `http://localhost:8501`, dengan fitur:
- Pilih kategori produk dan rentang tanggal
- Bandingkan kurva actual vs prediksi SARIMAX dan LSTM
- Lihat tabel perbandingan metrik (RMSE, MAE, MAPE) antar kategori
- Lihat rekomendasi model terbaik per kategori
- Eksplorasi tren penjualan historis multi-kategori

## Tools & Library

Python, Pandas, NumPy, Statsmodels (SARIMAX), TensorFlow/Keras (LSTM), Scikit-learn, Matplotlib, Seaborn, Streamlit, Plotly.

## Referensi

- Falatouri, T., et al. (2022). Predictive Analytics for Demand Forecasting — A Comparison of SARIMA and LSTM.
- Valles-Perez, I., et al. (2022). Approaching Sales Forecasting Using Recurrent Neural Networks and Transformers.
