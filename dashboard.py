import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ARTIFACT_DIR = Path('artifacts')

st.set_page_config(
    page_title='Retail Sales Forecasting Dashboard',
    page_icon='',
    layout='wide',
)

# ------------------------------------------------------------
# Loaders (cached supaya tidak reload setiap interaksi)
# ------------------------------------------------------------

@st.cache_data
def load_metadata():
    with open(ARTIFACT_DIR / 'metadata.json') as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    df = pd.read_parquet(ARTIFACT_DIR / 'predictions.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def load_metrics():
    df = pd.read_csv(ARTIFACT_DIR / 'metrics_comparison.csv', index_col=0)
    return df


@st.cache_data
def load_rekomendasi():
    return pd.read_csv(ARTIFACT_DIR / 'rekomendasi_model.csv', index_col=0)


@st.cache_data
def load_historical():
    df = pd.read_parquet(ARTIFACT_DIR / 'historical_sales.parquet')
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_resource
def load_sarimax_model(category, metadata):
    fname = ARTIFACT_DIR / 'sarimax_models' / f'sarimax_{category.replace(" ", "_")}.pkl'
    with open(fname, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_lstm_model(category):
    import tensorflow as tf
    fname = ARTIFACT_DIR / 'lstm_models' / f'lstm_{category.replace(" ", "_")}.keras'
    return tf.keras.models.load_model(fname)


@st.cache_resource
def load_lstm_scalers():
    with open(ARTIFACT_DIR / 'lstm_scalers.pkl', 'rb') as f:
        return pickle.load(f)


def check_artifacts_exist():
    required = [
        'predictions.parquet', 'metrics_comparison.csv',
        'rekomendasi_model.csv', 'historical_sales.parquet', 'metadata.json',
    ]
    missing = [r for r in required if not (ARTIFACT_DIR / r).exists()]
    return missing


# ------------------------------------------------------------
# Guard: cek artifacts tersedia
# ------------------------------------------------------------
missing = check_artifacts_exist()
if missing:
    st.error(
        'Folder **artifacts/** belum lengkap. File yang hilang:\n\n'
        + '\n'.join(f'- `{m}`' for m in missing)
        + '\n\nJalankan dulu cell "Simpan Artifacts" di notebook, lalu salin '
          'folder `artifacts/` ke folder yang sama dengan `dashboard.py` ini.'
    )
    st.stop()

metadata = load_metadata()
df_preds = load_predictions()
df_metrics = load_metrics()
df_rekomendasi = load_rekomendasi()
df_hist = load_historical()

CATEGORIES = metadata['selected_categories']

# ------------------------------------------------------------
# Sidebar — kontrol interaktif
# ------------------------------------------------------------
st.sidebar.title(' Kontrol Dashboard')

selected_category = st.sidebar.selectbox(
    'Pilih Kategori Produk', CATEGORIES, index=0
)

model_view = st.sidebar.radio(
    'Tampilkan Model', ['Keduanya', 'SARIMAX', 'LSTM'], index=0
)

date_min = df_preds['date'].min()
date_max = df_preds['date'].max()
date_range = st.sidebar.date_input(
    'Rentang Tanggal (data uji)',
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

show_history = st.sidebar.checkbox('Tampilkan data historis (sebelum cutoff)', value=False)

st.sidebar.markdown('---')
st.sidebar.caption(
    f"Cutoff train/test: **{metadata['cutoff_date']}**\n\n"
    f"Kategori dianalisis: **{len(CATEGORIES)}**"
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title('Retail Sales Forecasting Dashboard')
st.caption('Corporación Favorita — Perbandingan SARIMAX vs LSTM untuk prediksi penjualan harian per kategori produk')

# ------------------------------------------------------------
# Ringkasan metrik (kartu)
# ------------------------------------------------------------
row = df_metrics.loc[selected_category]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('SARIMAX — MAPE', f"{row['SARIMAX_MAPE']:.2f}%")
with col2:
    st.metric('LSTM — MAPE', f"{row['LSTM_MAPE']:.2f}%")
with col3:
    best = df_rekomendasi.loc[selected_category, 'Rekomendasi'] if 'Rekomendasi' in df_rekomendasi.columns else df_rekomendasi.loc[selected_category].iloc[0]
    st.metric('Model Direkomendasikan', str(best))
with col4:
    delta_rmse = row['SARIMAX_RMSE'] - row['LSTM_RMSE']
    st.metric('Selisih RMSE (SARIMAX − LSTM)', f"{delta_rmse:,.0f}")

st.markdown('---')

# ------------------------------------------------------------
# Tab utama
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    'Actual vs Prediksi', 'Perbandingan Metrik', 'Rekomendasi per Kategori', 'Tren Historis'
])

# --- TAB 1: Actual vs Predicted ---
with tab1:
    st.subheader(f'Actual vs Prediksi — {selected_category}')

    plot_df = df_preds[df_preds['family'] == selected_category].copy()

    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        plot_df = plot_df[(plot_df['date'] >= start) & (plot_df['date'] <= end)]

    fig = go.Figure()

    actual_df = plot_df[plot_df['model'] == 'SARIMAX'][['date', 'y_actual']].drop_duplicates()
    fig.add_trace(go.Scatter(
        x=actual_df['date'], y=actual_df['y_actual'],
        mode='lines', name='Actual', line=dict(color='#333333', width=2)
    ))

    if model_view in ('Keduanya', 'SARIMAX'):
        sub = plot_df[plot_df['model'] == 'SARIMAX']
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['y_pred'],
            mode='lines', name='SARIMAX', line=dict(color='#e84c4c', width=1.5, dash='dot')
        ))

    if model_view in ('Keduanya', 'LSTM'):
        sub = plot_df[plot_df['model'] == 'LSTM']
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['y_pred'],
            mode='lines', name='LSTM', line=dict(color='#4c72e8', width=1.5, dash='dot')
        ))

    fig.update_layout(
        height=480, xaxis_title='Tanggal', yaxis_title='Total Penjualan',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander('Lihat tabel data prediksi'):
        st.dataframe(plot_df.sort_values('date'), use_container_width=True)

# --- TAB 2: Perbandingan Metrik ---
with tab2:
    st.subheader('Perbandingan Metrik Evaluasi — Semua Kategori')

    metric_choice = st.radio('Pilih metrik', ['RMSE', 'MAE', 'MAPE'], horizontal=True)
    col_s, col_l = f'SARIMAX_{metric_choice}', f'LSTM_{metric_choice}'

    bar_df = df_metrics[[col_s, col_l]].reset_index().melt(
        id_vars='Kategori' if 'Kategori' in df_metrics.reset_index().columns else df_metrics.index.name or 'index',
        var_name='Model', value_name=metric_choice
    )
    bar_df['Model'] = bar_df['Model'].str.replace(f'_{metric_choice}', '', regex=False)

    fig2 = px.bar(
        bar_df, x=bar_df.columns[0], y=metric_choice, color='Model', barmode='group',
        color_discrete_map={'SARIMAX': '#e84c4c', 'LSTM': '#4c72e8'},
    )
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('**Tabel lengkap:**')
    st.dataframe(df_metrics.style.format('{:.2f}'), use_container_width=True)

    avg_sarimax_mape = df_metrics['SARIMAX_MAPE'].mean()
    avg_lstm_mape = df_metrics['LSTM_MAPE'].mean()
    st.info(
        f"Rata-rata MAPE seluruh kategori — SARIMAX: **{avg_sarimax_mape:.2f}%**, "
        f"LSTM: **{avg_lstm_mape:.2f}%**. Performa tidak seragam antar kategori; "
        f"lihat tab Rekomendasi untuk model terbaik per kategori."
    )

# --- TAB 3: Rekomendasi ---
with tab3:
    st.subheader('Rekomendasi Model per Kategori')
    st.caption('Ditentukan berdasarkan voting mayoritas dari 3 metrik (RMSE, MAE, MAPE)')
    st.dataframe(df_rekomendasi, use_container_width=True)

    sarimax_wins = (df_rekomendasi.astype(str).apply(lambda r: r.str.contains('SARIMAX').any(), axis=1)).sum()
    lstm_wins = (df_rekomendasi.astype(str).apply(lambda r: r.str.contains('LSTM').any(), axis=1)).sum()

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric('Kategori dimenangkan SARIMAX', int(sarimax_wins))
    with col_b:
        st.metric('Kategori dimenangkan LSTM', int(lstm_wins))

# --- TAB 4: Tren Historis ---
with tab4:
    st.subheader('Tren Penjualan Historis')
    hist_cats = st.multiselect('Pilih kategori untuk dibandingkan', CATEGORIES, default=[selected_category])

    if hist_cats:
        hist_plot = df_hist[df_hist['family'].isin(hist_cats)]
        if not show_history:
            cutoff = pd.to_datetime(metadata['cutoff_date'])
            hist_plot = hist_plot[hist_plot['date'] >= cutoff]

        fig3 = px.line(
            hist_plot, x='date', y='sales', color='family',
            labels={'sales': 'Total Penjualan', 'date': 'Tanggal', 'family': 'Kategori'},
        )
        fig3.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning('Pilih minimal satu kategori.')

st.markdown('---')
st.caption(
    'Dashboard ini menampilkan hasil prediksi yang sudah dihitung sebelumnya di notebook '
    '(tidak melatih ulang model secara live). Untuk memuat ulang hasil terbaru, jalankan '
    'ulang cell "Simpan Artifacts" di notebook lalu refresh folder artifacts/.'
)
