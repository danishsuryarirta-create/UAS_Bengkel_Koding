import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ===================================================================
# KONFIGURASI HALAMAN
# ===================================================================
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📉",
    layout="wide"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
IMPORTANCE_THRESHOLD = 0.005
DATA_PATH_CANDIDATES = [
    "Sales - Marketing customer dataset.csv",
    "Sales_-_Marketing_customer_dataset.csv",
    "Sales-Marketing_customer_dataset.csv",
]
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "selected_features.pkl"


# ===================================================================
# PIPELINE PREPROCESSING (mengikuti notebook BAGIAN 3)
# ===================================================================
def preprocess(df_raw: pd.DataFrame):
    df_prep = df_raw.copy()

    # Drop ID & kolom datetime (di luar scope feature engineering)
    df_prep.drop(columns=['customer_id', 'signup_date', 'last_purchase_date'],
                 inplace=True, errors='ignore')

    # coupon_code -> fitur biner has_coupon
    df_prep['has_coupon'] = df_prep['coupon_code'].notna().astype(int)
    df_prep.drop(columns=['coupon_code'], inplace=True, errors='ignore')

    # Imputasi numerik dengan median
    for col in ['age', 'total_spent', 'satisfaction_score']:
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].fillna(df_prep[col].median())

    # Imputasi kategorik dengan modus
    if 'gender' in df_prep.columns:
        df_prep['gender'] = df_prep['gender'].fillna(df_prep['gender'].mode()[0])

    # Hapus duplikat
    df_prep.drop_duplicates(inplace=True)

    # IQR Capping pada fitur numerik kontinyu (bukan kolom biner)
    binary_cols = ['churn', 'is_premium_user', 'discount_used',
                   'refund_requested', 'has_coupon']
    cap_cols = df_prep.select_dtypes(include='number').columns.difference(binary_cols).tolist()

    for col in cap_cols:
        Q1, Q3 = df_prep[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_prep[col] = df_prep[col].clip(lower=lower, upper=upper)

    # Label Encoding untuk subscription_type (2 nilai)
    le = LabelEncoder()
    df_prep['subscription_type'] = le.fit_transform(df_prep['subscription_type'])
    subscription_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # One-Hot Encoding untuk kolom nominal
    ohe_cols = ['gender', 'country', 'acquisition_channel', 'device_type', 'payment_method']
    df_prep = pd.get_dummies(df_prep, columns=ohe_cols, drop_first=True)

    # Drop city (kardinalitas tinggi)
    df_prep.drop(columns=['city'], inplace=True, errors='ignore')

    # Konversi kolom boolean hasil OHE ke int
    bool_cols = df_prep.select_dtypes(include='bool').columns
    df_prep[bool_cols] = df_prep[bool_cols].astype(int)

    return df_prep, subscription_mapping


@st.cache_resource(show_spinner="Melatih model (dijalankan sekali, hasil di-cache)...")
def load_or_train_artifacts():
    """
    Memuat model, scaler, dan daftar fitur dari file .pkl jika tersedia.
    Jika tidak tersedia, latih ulang mengikuti pipeline notebook
    (EDA -> Preprocessing -> Feature Selection -> Random Forest)
    lalu simpan hasilnya agar run selanjutnya lebih cepat.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(FEATURES_PATH)
        ohe_columns = joblib.load("ohe_columns.pkl") if os.path.exists("ohe_columns.pkl") else None
        subscription_mapping = joblib.load("subscription_mapping.pkl") if os.path.exists("subscription_mapping.pkl") else None
        return model, scaler, selected_features, ohe_columns, subscription_mapping

    data_path = None
    for candidate in DATA_PATH_CANDIDATES:
        if os.path.exists(candidate):
            data_path = candidate
            break
    if data_path is None:
        raise FileNotFoundError(
            "Dataset CSV tidak ditemukan. Pastikan file dataset berada di folder "
            "yang sama dengan app.py, dengan nama salah satu dari: "
            f"{DATA_PATH_CANDIDATES}"
        )
    df_raw = pd.read_csv(data_path)
    df_prep, subscription_mapping = preprocess(df_raw)

    X_prep = df_prep.drop(columns=['churn'])
    y_prep = df_prep['churn']
    ohe_columns = X_prep.columns.tolist()

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_prep, y_prep, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_prep
    )

    scaler_full = StandardScaler()
    X_train_sc = scaler_full.fit_transform(X_train_p)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train_p)

    rf_full = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_full.fit(X_train_sm, y_train_sm)

    # Feature selection berdasarkan feature importance
    fi_df = pd.DataFrame({
        'feature': X_prep.columns.tolist(),
        'importance': rf_full.feature_importances_
    }).sort_values('importance', ascending=False)

    selected_features = fi_df[fi_df['importance'] >= IMPORTANCE_THRESHOLD]['feature'].tolist()

    X_train_sel = X_train_p[selected_features]
    X_test_sel = X_test_p[selected_features]

    scaler = StandardScaler()
    X_train_sel_sc = scaler.fit_transform(X_train_sel)
    X_test_sel_sc = scaler.transform(X_test_sel)

    X_train_sel_sm, y_train_sel_sm = smote.fit_resample(X_train_sel_sc, y_train_p)

    model = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_sel_sm, y_train_sel_sm)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selected_features, FEATURES_PATH)
    joblib.dump(ohe_columns, "ohe_columns.pkl")
    joblib.dump(subscription_mapping, "subscription_mapping.pkl")

    return model, scaler, selected_features, ohe_columns, subscription_mapping


def build_feature_row(input_dict, ohe_columns, subscription_mapping):
    """Mengubah input form menjadi 1 baris dataframe sesuai struktur fitur training."""
    row = {col: 0 for col in ohe_columns}

    # Fitur numerik & biner langsung
    direct_fields = [
        'age', 'is_premium_user', 'total_visits', 'avg_session_time',
        'pages_per_session', 'email_open_rate', 'email_click_rate',
        'total_spent', 'avg_order_value', 'discount_used', 'support_tickets',
        'refund_requested', 'delivery_delay_days', 'satisfaction_score',
        'nps_score', 'marketing_spend_per_user', 'lifetime_value',
        'last_3_month_purchase_freq', 'has_coupon'
    ]
    for f in direct_fields:
        if f in row:
            row[f] = input_dict[f]

    # subscription_type via label encoding
    if 'subscription_type' in row:
        row['subscription_type'] = subscription_mapping.get(input_dict['subscription_type'], 0)

    # One-hot encoded categorical columns
    ohe_map = {
        'gender': input_dict['gender'],
        'country': input_dict['country'],
        'acquisition_channel': input_dict['acquisition_channel'],
        'device_type': input_dict['device_type'],
        'payment_method': input_dict['payment_method'],
    }
    for prefix, value in ohe_map.items():
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1
        # Jika value adalah kategori baseline (drop_first), semua kolom OHE tetap 0 -> sudah benar

    return pd.DataFrame([row])[ohe_columns]


# ===================================================================
# LOAD MODEL & ARTIFACTS
# ===================================================================
model, scaler, selected_features, ohe_columns, subscription_mapping = load_or_train_artifacts()

if subscription_mapping is None:
    subscription_mapping = {'Annual': 0, 'Monthly': 1}

# ===================================================================
# UI
# ===================================================================
st.title("📉 Prediksi Churn Pelanggan")
st.caption("UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro (UDINUS)")
st.markdown(
    "Aplikasi ini memprediksi kemungkinan seorang pelanggan akan **churn** (berhenti "
    "menggunakan layanan) berdasarkan data perilaku, transaksi, dan demografi pelanggan."
)

tab_predict, tab_about = st.tabs(["🔮 Prediksi", "ℹ️ Tentang Model"])

with tab_predict:
    st.subheader("Masukkan Data Pelanggan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demografi & Akun**")
        age = st.number_input("Usia", min_value=15, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        country = st.selectbox("Negara", ["Bangladesh", "Germany", "India", "UK", "USA"])
        subscription_type = st.selectbox("Tipe Subscription", ["Annual", "Monthly"])
        is_premium_user = st.selectbox("Pengguna Premium?", ["Tidak", "Ya"])
        device_type = st.selectbox("Tipe Device", ["Desktop", "Mobile", "Tablet"])
        acquisition_channel = st.selectbox(
            "Channel Akuisisi", ["Email", "Facebook Ads", "Google Ads", "Organic", "Referral"]
        )

    with col2:
        st.markdown("**Aktivitas & Engagement**")
        total_visits = st.number_input("Total Kunjungan", min_value=0, max_value=100, value=15)
        avg_session_time = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0, value=8.0)
        pages_per_session = st.number_input("Halaman per Sesi", min_value=0.0, value=4.0)
        email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
        email_click_rate = st.slider("Email Click Rate", 0.0, 1.0, 0.25)
        nps_score = st.slider("NPS Score", 0, 10, 5)
        satisfaction_score = st.slider("Satisfaction Score", 1.0, 5.0, 3.5)

    with col3:
        st.markdown("**Transaksi & Layanan**")
        total_spent = st.number_input("Total Belanja", min_value=0.0, value=500.0)
        avg_order_value = st.number_input("Rata-rata Nilai Order", min_value=0.0, value=60.0)
        lifetime_value = st.number_input("Lifetime Value", min_value=0.0, value=1200.0)
        marketing_spend_per_user = st.number_input("Marketing Spend per User", min_value=0.0, value=17.0)
        last_3_month_purchase_freq = st.number_input("Frekuensi Pembelian 3 Bulan Terakhir", min_value=0, value=7)
        discount_used = st.selectbox("Pernah Pakai Diskon?", ["Tidak", "Ya"])
        has_coupon = st.selectbox("Pernah Pakai Kupon?", ["Tidak", "Ya"])
        payment_method = st.selectbox("Metode Pembayaran", ["BKash", "Card", "PayPal", "SEPA", "UPI"])
        support_tickets = st.number_input("Jumlah Tiket Support", min_value=0, value=2)
        refund_requested = st.selectbox("Pernah Minta Refund?", ["Tidak", "Ya"])
        delivery_delay_days = st.number_input("Hari Keterlambatan Pengiriman", min_value=0, value=3)

    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Churn", type="primary", use_container_width=True)

    if predict_btn:
        input_dict = {
            'age': age,
            'gender': gender,
            'country': country,
            'subscription_type': subscription_type,
            'is_premium_user': 1 if is_premium_user == "Ya" else 0,
            'device_type': device_type,
            'acquisition_channel': acquisition_channel,
            'total_visits': total_visits,
            'avg_session_time': avg_session_time,
            'pages_per_session': pages_per_session,
            'email_open_rate': email_open_rate,
            'email_click_rate': email_click_rate,
            'nps_score': nps_score,
            'satisfaction_score': satisfaction_score,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'lifetime_value': lifetime_value,
            'marketing_spend_per_user': marketing_spend_per_user,
            'last_3_month_purchase_freq': last_3_month_purchase_freq,
            'discount_used': 1 if discount_used == "Ya" else 0,
            'has_coupon': 1 if has_coupon == "Ya" else 0,
            'payment_method': payment_method,
            'support_tickets': support_tickets,
            'refund_requested': 1 if refund_requested == "Ya" else 0,
            'delivery_delay_days': delivery_delay_days,
        }

        full_row = build_feature_row(input_dict, ohe_columns, subscription_mapping)
        selected_row = full_row[selected_features]
        scaled_row = scaler.transform(selected_row)

        pred = model.predict(scaled_row)[0]
        proba = model.predict_proba(scaled_row)[0][1]

        st.markdown("### 🎯 Hasil Prediksi")
        c1, c2 = st.columns(2)
        with c1:
            if pred == 1:
                st.error(f"⚠️ Pelanggan **diprediksi CHURN**")
            else:
                st.success(f"✅ Pelanggan **diprediksi TIDAK CHURN**")
        with c2:
            st.metric("Probabilitas Churn", f"{proba*100:.1f}%")

        st.progress(min(max(proba, 0.0), 1.0))

        if proba >= 0.7:
            st.warning("Risiko churn **tinggi** — pertimbangkan tindakan retensi segera (diskon, follow-up personal, dsb).")
        elif proba >= 0.4:
            st.info("Risiko churn **sedang** — pantau aktivitas pelanggan secara berkala.")
        else:
            st.info("Risiko churn **rendah** — pelanggan tampak stabil.")

with tab_about:
    st.subheader("Tentang Model")
    st.markdown(
        """
        Model ini dilatih mengikuti pipeline dari notebook **UAS Bengkel Koding Data Science**:

        1. **EDA** — eksplorasi missing value, distribusi churn, korelasi fitur.
        2. **Preprocessing** — imputasi median/modus, fitur biner `has_coupon`,
           IQR capping untuk outlier, Label Encoding (`subscription_type`),
           One-Hot Encoding (gender, country, acquisition_channel, device_type, payment_method),
           drop kolom `city`, `customer_id`, dan kolom tanggal.
        3. **Scaling & SMOTE** — `StandardScaler` (fit pada train saja) dan `SMOTE`
           untuk menangani ketidakseimbangan kelas churn pada data train.
        4. **Feature Selection** — fitur dipilih berdasarkan *feature importance*
           dari Random Forest dengan threshold ≥ 0.005.
        5. **Model Final** — Random Forest Classifier, dioptimalkan dengan
           skema yang sama seperti pada notebook (hyperparameter tuning via
           `RandomizedSearchCV` + `StratifiedKFold`).

        Metrik evaluasi yang digunakan: **Accuracy, Precision, Recall, F1-Score**,
        dengan F1-Score sebagai metrik utama karena *False Negative* (pelanggan
        churn yang tidak terdeteksi) lebih mahal secara bisnis dibanding *False Positive*.
        """
    )

    st.markdown("**Fitur yang digunakan model (setelah feature selection):**")
    st.write(selected_features)
