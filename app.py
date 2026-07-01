import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. KONFIGURASI HALAMAN & UTILITY THEME
# ==============================================================================
st.set_page_config(
    page_title="Insightify Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk mempercantik UI Dashboard
st.markdown("""
    <style>
    .main-header {
        font-size:36px !important;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size:18px !important;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama Aplikasi
st.markdown('<div class="main-header">🔮 Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">UAS Bengkel Koding Data Science | Danish Suryarirta Kusuma Yudha (A11.2023.14873)</div>', unsafe_allow_html=True)

# ==============================================================================
# 2. HELPER FUNCTION: PREPROCESSING DATA RAW
# ==============================================================================
def preprocess_input(df_raw, selected_features=None, scaler=None):
    """
    Fungsi untuk mereplikasi seluruh tahapan preprocessing dari Notebook UAS:
    - Menghapus ID & Kolom Datetime
    - Penanganan Missing Value (Imputasi Median/Modus dan Rekayasa has_coupon)
    - One-Hot Encoding & Penyelarasan Kolom Fitur
    - Scaling menggunakan objek StandardScaler yang sudah dilatih
    """
    df_prep = df_raw.copy()
    
    # 1. Drop kolom yang tidak bisa digunakan langsung oleh model
    cols_to_drop = ['customer_id', 'signup_date', 'last_purchase_date', 'city']
    df_prep.drop(columns=[c for col in cols_to_drop if (c := col) in df_prep.columns], inplace=True, errors='ignore')
    
    # 2. Transformasi coupon_code -> has_coupon (Fitur biner)
    if 'coupon_code' in df_prep.columns:
        df_prep['has_coupon'] = df_prep['coupon_code'].notna().astype(int)
        df_prep.drop(columns=['coupon_code'], inplace=True)
    elif 'has_coupon' not in df_prep.columns:
        df_prep['has_coupon'] = 0

    # 3. Imputasi Missing Values dengan nilai median dari data training awal
    medians = {
        'age': 35.0, 'total_spent': 504.25, 'satisfaction_score': 4.0,
        'total_visits': 15.0, 'avg_session_time': 7.97, 'pages_per_session': 3.94,
        'email_open_rate': 0.5, 'email_click_rate': 0.25, 'avg_order_value': 60.61,
        'discount_used': 0, 'support_tickets': 2.0, 'refund_requested': 0,
        'delivery_delay_days': 3.0, 'nps_score': 5.0, 'marketing_spend_per_user': 17.47,
        'lifetime_value': 1199.70, 'last_3_month_purchase_freq': 7.0, 'is_premium_user': 0
    }
    for col, val in medians.items():
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].fillna(val)
            
    if 'gender' in df_prep.columns:
        df_prep['gender'] = df_prep['gender'].fillna('Female')

    # 4. Label Encoding Manual untuk subscription_type (Monthly: 1, Annual: 0)
    if 'subscription_type' in df_prep.columns:
        df_prep['subscription_type'] = df_prep['subscription_type'].map({'Monthly': 1, 'Annual': 0}).fillna(1)

    # 5. One-Hot Encoding untuk variabel nominal hasil kategori
    ohe_cols = ['gender', 'country', 'acquisition_channel', 'device_type', 'payment_method']
    df_prep = pd.get_dummies(df_prep, columns=[c for c in ohe_cols if c in df_prep.columns], drop_first=True)
    
    # Konversi tipe kolom boolean hasil OHE menjadi integer (0/1)
    bool_cols = df_prep.select_dtypes(include='bool').columns
    df_prep[bool_cols] = df_prep[bool_cols].astype(int)

    # 6. Menyelaraskan seluruh kolom dummies agar identik dengan saat pelatihan model
    all_training_cols = [
        'is_premium_user', 'total_visits', 'avg_session_time', 'pages_per_session',
        'email_open_rate', 'email_click_rate', 'total_spent', 'avg_order_value',
        'discount_used', 'support_tickets', 'refund_requested', 'delivery_delay_days',
        'satisfaction_score', 'nps_score', 'marketing_spend_per_user', 'lifetime_value',
        'last_3_month_purchase_freq', 'age', 'has_coupon', 'subscription_type',
        'gender_Male', 'gender_Other', 'country_Germany', 'country_India', 'country_UK', 'country_USA',
        'acquisition_channel_Facebook Ads', 'acquisition_channel_Google Ads', 
        'acquisition_channel_Organic', 'acquisition_channel_Referral',
        'device_type_Mobile', 'device_type_Tablet', 
        'payment_method_Credit Card', 'payment_method_PayPal', 'payment_method_SEPA', 'payment_method_UPI'
    ]
    
    # Tambahkan kolom dummy yang tidak ada di data baru dengan nilai default 0
    for col in all_training_cols:
        if col not in df_prep.columns:
            df_prep[col] = 0
            
    # Susun ulang urutan kolom
    df_prep = df_prep[all_training_cols]

    # 7. Memilih fitur terbaik (Feature Selection) dan penskalaan (Scaling)
    if selected_features is not None:
        df_prep = df_prep[selected_features]
        
    if scaler is not None:
        X_scaled = scaler.transform(df_prep)
        return X_scaled
        
    return df_prep

# ==============================================================================
# 3. MEMUAT MODUL ARTIFAK MODEL (.PKL)
# ==============================================================================
@st.cache_resource
def load_artifacts():
    """Memuat pkl hasil simpanan joblib dari notebook"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('selected_features.pkl')
        return model, scaler, features, True
    except FileNotFoundError:
        return None, None, None, False

model, scaler, selected_features, artifacts_loaded = load_artifacts()

# ==============================================================================
# 4. NAVIGASI SIDEBAR
# ==============================================================================
st.sidebar.title("🧭 Menu Navigasi")
menu = st.sidebar.radio("Pilih Mode Prediksi:", ["Prediksi Individu", "Prediksi Batch (CSV File)"])

# Jika file pkl tidak ditemukan, tetap tampilkan peringatan tersembunyi di sidebar tanpa metrik informasi
if not artifacts_loaded:
    st.sidebar.warning("⚠️ Berkas `.pkl` tidak ditemukan. Aplikasi berjalan dalam Mode Demo.")

# ==============================================================================
# 5. MODE 1: PREDIKSI INDIVIDU VIA FORMULIR
# ==============================================================================
if menu == "Prediksi Individu":
    st.subheader("📋 Input Data Profil Pelanggan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### **Karakteristik & Akun**")
        age = st.number_input("Usia (Age)", min_value=1, max_value=100, value=35)
        gender = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
        country = st.selectbox("Negara Asal", ["USA", "Germany", "India", "UK", "France"])
        subscription_type = st.selectbox("Tipe Langganan", ["Monthly", "Annual"])
        is_premium_user = st.selectbox("Pengguna Premium?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        has_coupon = st.selectbox("Menggunakan Kupon?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")

    with col2:
        st.markdown("##### **Metrik Aktivitas & Penggunaan**")
        total_visits = st.number_input("Total Kunjungan Sesi", min_value=0, value=15)
        avg_session_time = st.number_input("Rata-rata Durasi Sesi (Menit)", min_value=0.0, value=7.9, step=0.1)
        pages_per_session = st.number_input("Halaman yang Dibuka per Sesi", min_value=0.0, value=3.9, step=0.1)
        total_spent = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=504.0, step=10.0)
        avg_order_value = st.number_input("Rata-rata Nilai Order ($)", min_value=0.0, value=60.0, step=5.0)
        last_3_month_freq = st.number_input("Frekuensi Belanja (3 Bulan Terakhir)", min_value=0, value=7)

    with col3:
        st.markdown("##### **Tingkat Kepuasan & Kendala**")
        satisfaction_score = st.slider("Skor Kepuasan (1-5)", 1.0, 5.0, 4.0, step=0.5)
        nps_score = st.slider("Skor NPS Net Promoter (0-10)", 0, 10, 5)
        support_tickets = st.number_input("Jumlah Pengaduan Tiket Masalah", min_value=0, value=2)
        delivery_delay_days = st.number_input("Hari Keterlambatan Pengiriman", min_value=0, value=3)
        discount_used = st.selectbox("Pernah Menggunakan Diskon?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        refund_requested = st.selectbox("Pernah Mengajukan Refund?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        marketing_spend = st.number_input("Biaya Pemasaran per Pengguna ($)", min_value=0.0, value=17.5)
        lifetime_value = st.number_input("Customer Lifetime Value (CLV)", min_value=0.0, value=1200.0)
        acquisition_channel = st.selectbox("Saluran Akuisisi", ["Organic", "Facebook Ads", "Google Ads", "Referral"])
        payment_method = st.selectbox("Metode Pembayaran", ["Credit Card", "PayPal", "SEPA", "UPI"])

    st.markdown("---")
    
    if st.button("🚀 Prediksi Status Churn", type="primary"):
        # Tampung input ke dalam struktur DataFrame data tunggal
        input_dict = {
            'age': age, 'gender': gender, 'country': country, 'subscription_type': subscription_type,
            'is_premium_user': is_premium_user, 'has_coupon': has_coupon, 'total_visits': total_visits,
            'avg_session_time': avg_session_time, 'pages_per_session': pages_per_session,
            'total_spent': total_spent, 'avg_order_value': avg_order_value, 
            'last_3_month_purchase_freq': last_3_month_freq, 'satisfaction_score': satisfaction_score,
            'nps_score': nps_score, 'support_tickets': support_tickets, 'delivery_delay_days': delivery_delay_days,
            'discount_used': discount_used, 'refund_requested': refund_requested,
            'marketing_spend_per_user': marketing_spend, 'lifetime_value': lifetime_value,
            'acquisition_channel': acquisition_channel, 'payment_method': payment_method
        }
        df_input = pd.DataFrame([input_dict])
        
        # Eksekusi Prediksi
        if artifacts_loaded:
            X_proc = preprocess_input(df_input, selected_features=selected_features, scaler=scaler)
            prediction = model.predict(X_proc)[0]
            probability = model.predict_proba(X_proc)[0][1] if hasattr(model, "predict_proba") else None
        else:
            # Simulasi fallback pintar berbasis korelasi logika sederhana jika .pkl absen
            probability = 0.15
            if support_tickets >= 3: probability += 0.35
            if satisfaction_score <= 2.5: probability += 0.30
            if delivery_delay_days > 4: probability += 0.15
            probability = min(probability, 0.98)
            prediction = 1 if probability >= 0.5 else 0

        # Output Tampilan Hasil
        st.subheader("📊 Hasil Analisis Prediksi")
        c_res1, c_res2 = st.columns([1, 2])
        
        with c_res1:
            if prediction == 1:
                st.error("🚨 POTENSI CHURN (RISIKO TINGGI)")
            else:
                st.success("✅ SETIA / LOYAL (RISIKO RENDAH)")
                
            if probability is not None:
                st.metric(label="Probabilitas Churn Pelanggan", value=f"{probability*100:.2f}%")
        
        with c_res2:
            st.markdown("**💡 Rekomendasi Tindakan Operasional Bisnis:**")
            if prediction == 1:
                st.write("- 📞 **Prioritas Hubungi:** Tim sales perlu segera berinteraksi kembali guna menanyakan kepuasan.")
                st.write("- 🛠️ **Evaluasi Tiket Masalah:** Segera selesaikan sisa pengaduan tiket komplain user.")
                st.write("- 🎁 **Insentif Khusus:** Kirimkan voucher penawaran personal/diskon khusus.")
            else:
                st.write("- 🌟 **Menjaga Loyalitas:** Tetap pertahaman pelayanan prima demi kepuasan berkelanjutan.")
                st.write("- 🚀 **Rekomendasi Up-selling:** Tawarkan promosi peningkatan paket langganan ke tipe tahunan (*Annual*).")

# ==============================================================================
# 6. MODE 2: PREDIKSI MASSAL / BATCH VIA UNGGAH CSV
# ==============================================================================
elif menu == "Prediksi Batch (CSV File)":
    st.subheader("📁 Prediksi Massal Pelanggan via CSV")
    st.write("Unggah dokumen berkas data pelanggan berformat `.csv` untuk menganalisis risiko churn secara bersamaan.")
    
    uploaded_file = st.file_uploader("Pilih Berkas CSV Anda:", type=["csv"])
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"📂 Berhasil memuat data: **{df_batch.shape[0]} baris** dan **{df_batch.shape[1]} kolom**.")
        
        st.markdown("### 🔍 Cuplikan 5 Data Teratas yang Diunggah")
        st.dataframe(df_batch.head(5))
        
        if st.button("⚙️ Jalankan Prediksi Massal", type="primary"):
            with st.spinner("Sedang memproses penyelarasan data dan inferensi model..."):
                try:
                    if artifacts_loaded:
                        X_batch_proc = preprocess_input(df_batch, selected_features=selected_features, scaler=scaler)
                        preds = model.predict(X_batch_proc)
                        probs = model.predict_proba(X_batch_proc)[:, 1] if hasattr(model, "predict_proba") else [0.0]*len(preds)
                    else:
                        # Simulasi acak terarah jika pkl tidak ada
                        np.random.seed(42)
                        probs = np.random.uniform(0.05, 0.85, size=len(df_batch))
                        preds = (probs >= 0.5).astype(int)
                    
                    # Tambahkan matriks hasil ke dataframe asli
                    df_batch['Prediksi_Churn'] = preds
                    df_batch['Probabilitas_Churn'] = probs
                    df_batch['Status_Pelanggan'] = df_batch['Prediksi_Churn'].map({1: 'Potensi Churn', 0: 'Setia (Loyal)'})
                    
                    st.success("✅ Analisis prediksi massal selesai dilakukan!")
                    
                    # Menghitung ringkasan data hasil prediksi
                    total_pelanggan = len(df_batch)
                    churn_count = int((df_batch['Prediksi_Churn'] == 1).sum())
                    loyal_count = int((df_batch['Prediksi_Churn'] == 0).sum())
                    churn_rate = (churn_count / total_pelanggan) * 100 if total_pelanggan > 0 else 0

                    # Menampilkan metrik ringkasan menggunakan kolom Streamlit
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-box"><b>Total Pelanggan Diuji</b><br/><span style="font-size:24px; color:#1E3A8A;">{total_pelanggan:,}</span></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-box" style="border-left: 5px solid #EF4444;"><b>🔴 Diprediksi Churn</b><br/><span style="font-size:24px; color:#DC2626; font-weight:bold;">{churn_count:,} ({churn_rate:.1f}%)</span></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-box" style="border-left: 5px solid #10B981;"><b>🟢 Diprediksi Setia (Loyal)</b><br/><span style="font-size:24px; color:#059669; font-weight:bold;">{loyal_count:,}</span></div>', unsafe_allow_html=True)

                    # Tampilkan visualisasi diagram distribusi sederhana hasil prediksi batch
                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors = ['#2ecc71', '#e74c3c']
                    counts = [loyal_count, churn_count]
                    labels = ['Setia (0)', 'Churn (1)']
                    ax.bar(labels, counts, color=colors, width=0.4)
                    ax.set_title("Proporsi Prediksi Kelas Churn Pelanggan", fontsize=10, fontweight='bold')
                    ax.set_ylabel("Jumlah Pelanggan")
                    st.pyplot(fig)

                    st.markdown("### 📈 Tabel Hasil Analisis Lengkap (Menampilkan 50 Baris Pertama)")
                    
                    # Tampilkan data hasil prediksi dalam bentuk dataframe interaktif
                    display_cols = ['customer_id', 'Status_Pelanggan', 'Probabilitas_Churn'] if 'customer_id' in df_batch.columns else ['Status_Pelanggan', 'Probabilitas_Churn']
                    remaining_cols = [c for c in df_batch.columns if c not in display_cols]
                    
                    # Gunakan seluruh fitur latih sebagai visualisasi tabel jika model termuat
                    st.dataframe(df_batch[display_cols + remaining_cols].head(50))
                    
                    # Sediakan tombol download berkas hasil prediksi baru
                    csv_data = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi Lengkap (.CSV)",
                        data=csv_data,
                        file_name="hasil_churn_prediction_uas.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memproses file CSV: {str(e)}")
