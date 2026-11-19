"""
Streamlit GUI — First-Order Sugeno FIS untuk peramalan beban listrik Jerman.

Jalankan:
    pip install -r requirements.txt
    streamlit run app.py

Dataset yang digunakan:
- utc_timestamp
- DE_load_actual_entsoe_transparency

Aplikasi juga menerima CSV lain jika memiliki dua kolom tersebut.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sugeno FIS - Electricity Load Forecasting",
    page_icon="⚡",
    layout="wide",
)

TIMESTAMP_COL = "utc_timestamp"
TARGET_COL = "DE_load_actual_entsoe_transparency"
FEATURE_COLS = ["X1_load_lag_1_interval", "X2_load_lag_24h"]
MF_NAMES = ["Triangular", "Trapezoidal", "Gaussian", "Generalized Bell", "Sigmoid"]
LINGUISTIC_LABELS = ["Low", "Medium", "High"]
RULE_COMBINATIONS = [(i, j) for i in range(3) for j in range(3)]
RULE_NAMES = [
    f"R{k+1}: {LINGUISTIC_LABELS[i]}-{LINGUISTIC_LABELS[j]}"
    for k, (i, j) in enumerate(RULE_COMBINATIONS)
]

DEFAULT_DATA_PATH = (
    "/kaggle/input/datasets/alharist/entso-full-exo-merged/"
    "dataset/df_DE.csv"
)


# ---------------------------------------------------------------------
# Fungsi membership
# ---------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x - a) / (b - a)
    right = np.ones_like(x) if b == c else (c - x) / (c - b)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x - a) / (b - a)
    right = np.ones_like(x) if c == d else (d - x) / (d - c)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0.0, 1.0)


def get_membership_parameters(x, q_low=10, q_high=90):
    """Parameter batas dan pusat berdasarkan data latih."""
    x = np.asarray(x, dtype=float)
    lo, q1, q2, q3, hi = np.quantile(
        x, [0.0, q_low / 100.0, 0.50, q_high / 100.0, 1.0]
    )
    eps = max(abs(hi - lo) * 1e-6, 1e-9)
    if q1 <= lo:
        q1 = lo + eps
    if q2 <= q1:
        q2 = q1 + eps
    if q3 <= q2:
        q3 = q2 + eps
    if hi <= q3:
        hi = q3 + eps
    return (float(lo), float(q1), float(q2), float(q3), float(hi))


def raw_memberships(x, mf_name, params, width_scale=1.0):
    """Menghasilkan membership Low, Medium, High; dinormalisasi per baris."""
    x = np.asarray(x, dtype=float).reshape(-1)
    lo, q1, q2, q3, hi = params

    span_left = max(q2 - q1, (hi - lo) * 1e-3, 1e-9)
    span_right = max(q3 - q2, (hi - lo) * 1e-3, 1e-9)
    x = np.clip(x, lo, hi)
    scale = max(float(width_scale), 1e-4)

    if mf_name == "Triangular":
        mus = np.column_stack([
            trimf(x, lo, lo, q2),
            trimf(x, q1, q2, q3),
            trimf(x, q2, hi, hi),
        ])
    elif mf_name == "Trapezoidal":
        mus = np.column_stack([
            trapmf(x, lo, lo, q1, q2),
            trapmf(
                x,
                q1,
                q1 + 0.25 * (q2 - q1),
                q3 - 0.25 * (q3 - q2),
                q3,
            ),
            trapmf(x, q2, q3, hi, hi),
        ])
    elif mf_name == "Gaussian":
        centers = [q1, q2, q3]
        widths = [span_left, (span_left + span_right) / 2, span_right]
        mus = np.column_stack([
            np.exp(-0.5 * ((x - c) / max(s * scale, 1e-9)) ** 2)
            for c, s in zip(centers, widths)
        ])
    elif mf_name == "Generalized Bell":
        centers = [q1, q2, q3]
        widths = [span_left, (span_left + span_right) / 2, span_right]
        bell_b = 2.0
        mus = np.column_stack([
            1.0 / (
                1.0 + np.abs((x - c) / max(s * scale, 1e-9)) ** (2 * bell_b)
            )
            for c, s in zip(centers, widths)
        ])
    elif mf_name == "Sigmoid":
        k_left = 4.0 / max(span_left * scale, 1e-9)
        k_right = 4.0 / max(span_right * scale, 1e-9)
        low = sigmoid(-k_left * (x - q1))
        rising = sigmoid(k_left * (x - q1))
        falling = sigmoid(-k_right * (x - q3))
        high = sigmoid(k_right * (x - q3))
        medium = np.minimum(rising, falling)
        mus = np.column_stack([low, medium, high])
    else:
        raise ValueError(f"Jenis membership tidak dikenal: {mf_name}")

    # Normalisasi agar Low + Medium + High = 1 untuk setiap input.
    mus = np.clip(mus, 0.0, None)
    return mus / np.maximum(mus.sum(axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------
# First-Order Sugeno FIS
# ---------------------------------------------------------------------
class SugenoFIS:
    def __init__(
        self,
        mf_name="Triangular",
        width_scale=1.0,
        ridge_alpha=1e-5,
        q_low=10,
        q_high=90,
    ):
        self.mf_name = mf_name
        self.width_scale = width_scale
        self.ridge_alpha = ridge_alpha
        self.q_low = q_low
        self.q_high = q_high

    def memberships(self, X):
        X = np.asarray(X, dtype=float)
        m1 = raw_memberships(
            X[:, 0], self.mf_name, self.params_x1, self.width_scale
        )
        m2 = raw_memberships(
            X[:, 1], self.mf_name, self.params_x2, self.width_scale
        )
        return m1, m2

    def firing_strengths(self, X):
        m1, m2 = self.memberships(X)
        weights = np.column_stack([
            m1[:, i] * m2[:, j] for i, j in RULE_COMBINATIONS
        ])
        return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(X) < 10:
            raise ValueError("Data latih terlalu sedikit (minimal 10 baris).")

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std < 1e-12] = 1.0
        X_scaled = (X - self.x_mean) / self.x_std

        self.params_x1 = get_membership_parameters(
            X[:, 0], self.q_low, self.q_high
        )
        self.params_x2 = get_membership_parameters(
            X[:, 1], self.q_low, self.q_high
        )

        weights = self.firing_strengths(X)
        design = np.column_stack([X_scaled, np.ones(len(X_scaled))])
        consequents = []

        for rule_idx in range(weights.shape[1]):
            w = weights[:, rule_idx]
            design_w = design * np.sqrt(w[:, None])
            target_w = y * np.sqrt(w)

            reg = self.ridge_alpha * np.eye(design.shape[1])
            reg[-1, -1] = 0.0  # Intersep tidak diregularisasi.

            beta = np.linalg.solve(
                design_w.T @ design_w + reg,
                design_w.T @ target_w,
            )
            consequents.append(beta)

        self.consequents = np.asarray(consequents)
        return self

    def explain_one(self, x1, x2):
        X = np.array([[float(x1), float(x2)]])
        m1, m2 = self.memberships(X)
        firing = self.firing_strengths(X)[0]

        X_scaled = (X - self.x_mean) / self.x_std
        design = np.array([X_scaled[0, 0], X_scaled[0, 1], 1.0])
        local_outputs = self.consequents @ design
        contributions = firing * local_outputs
        prediction = float(contributions.sum())

        membership_table = pd.DataFrame({
            "Himpunan": LINGUISTIC_LABELS,
            "X1 — lag 1 interval": m1[0],
            "X2 — lag 24 jam": m2[0],
        })

        rules_table = pd.DataFrame({
            "Aturan": RULE_NAMES,
            "Kekuatan aktivasi": firing,
            "Output lokal aturan": local_outputs,
            "Kontribusi tertimbang": contributions,
        })
        return prediction, membership_table, rules_table


# ---------------------------------------------------------------------
# Pemuatan data dan rekayasa fitur
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare_data(file_bytes, file_name):
    from io import BytesIO

    raw = pd.read_csv(BytesIO(file_bytes))
    if TIMESTAMP_COL not in raw.columns or TARGET_COL not in raw.columns:
        raise ValueError(
            f"CSV wajib memiliki kolom '{TIMESTAMP_COL}' dan '{TARGET_COL}'."
        )

    data = raw[[TIMESTAMP_COL, TARGET_COL]].copy()
    data[TIMESTAMP_COL] = pd.to_datetime(
        data[TIMESTAMP_COL], utc=True, errors="coerce"
    )
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = (
        data.dropna()
        .sort_values(TIMESTAMP_COL)
        .drop_duplicates(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    diffs = data[TIMESTAMP_COL].diff().dropna()
    if diffs.empty:
        raise ValueError("Data tidak cukup untuk menghitung interval sampling.")
    interval = diffs.mode().iloc[0]

    load_by_time = data.set_index(TIMESTAMP_COL)[TARGET_COL].sort_index()
    features = pd.DataFrame({TIMESTAMP_COL: data[TIMESTAMP_COL]})
    features["X1_load_lag_1_interval"] = (
        features[TIMESTAMP_COL] - interval
    ).map(load_by_time)
    features["X2_load_lag_24h"] = (
        features[TIMESTAMP_COL] - pd.Timedelta(hours=24)
    ).map(load_by_time)
    features["Y_load_current"] = features[TIMESTAMP_COL].map(load_by_time)

    features = features.dropna().reset_index(drop=True)
    if len(features) < 50:
        raise ValueError(
            "Observasi valid setelah pembentukan lag kurang dari 50. "
            "Periksa kelengkapan timestamp dan data."
        )
    return features, str(interval)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = np.abs(y_true) > 1e-12
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))
        * 100
        if nonzero.any()
        else np.nan
    )
    wape = (
        np.sum(np.abs(y_true - y_pred))
        / max(np.sum(np.abs(y_true)), 1e-12)
        * 100
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "WAPE (%)": wape}


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("⚡ First-Order Sugeno Fuzzy Inference System")
st.caption(
    "GUI peramalan beban listrik Jerman — pemilihan fungsi membership, "
    "tuning parameter, dan penjelasan proses inferensi."
)

with st.sidebar:
    st.header("1. Dataset")
    uploaded_file = st.file_uploader("Upload file df_DE.csv", type=["csv"])
    st.caption(
        "CSV harus memuat kolom utc_timestamp dan "
        "DE_load_actual_entsoe_transparency."
    )

    st.header("2. Konfigurasi Fuzzy")
    selected_mf = st.selectbox("Jenis membership function", MF_NAMES, index=0)
    width_scale = st.slider(
        "Width scale / kemiringan",
        min_value=0.50,
        max_value=2.00,
        value=1.00,
        step=0.05,
        help=(
            "Gaussian/Bell: mengubah lebar kurva. Sigmoid: mengubah "
            "kemiringan transisi. Triangular/Trapezoidal memakai bentuk "
            "kuantil tetap sehingga skala tidak mengubah bentuk dasarnya."
        ),
    )
    q_low = st.slider("Kuantil batas Low (%)", 1, 30, 10)
    q_high = st.slider("Kuantil batas High (%)", 70, 99, 90)
    ridge_alpha = st.number_input(
        "Ridge alpha konsekuen",
        min_value=0.0,
        max_value=1.0,
        value=0.00001,
        step=0.00001,
        format="%.5f",
    )
    train_fraction = st.slider(
        "Proporsi data latih kronologis",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
    )
    fit_button = st.button("Latih / perbarui model", type="primary", use_container_width=True)

# Pilih sumber data
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
else:
    st.info("Upload CSV pada sidebar untuk memulai. Pada Kaggle, Anda juga dapat memakai path dataset default.")
    default_path = st.text_input("Atau masukkan path CSV lokal", value=DEFAULT_DATA_PATH)
    if not Path(default_path).exists():
        st.warning("File belum ditemukan. Upload CSV atau periksa path dataset.")
        st.stop()
    file_bytes = Path(default_path).read_bytes()
    file_name = Path(default_path).name

try:
    features, detected_interval = load_and_prepare_data(file_bytes, file_name)
except Exception as exc:
    st.error(f"Gagal memuat data: {exc}")
    st.stop()

st.success(
    f"Data siap: {len(features):,} observasi fitur. "
    f"Interval sampling dominan: {detected_interval}."
)

# Model dilatih ulang saat tombol ditekan atau konfigurasi berubah.
config = (
    selected_mf, width_scale, q_low, q_high, ridge_alpha, train_fraction,
    len(features), file_name,
)
if "last_config" not in st.session_state:
    st.session_state.last_config = None
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None

if fit_button or st.session_state.last_config != config:
    with st.spinner("Melatih model Sugeno FIS..."):
        split_idx = int(len(features) * train_fraction)
        train_data = features.iloc[:split_idx].copy()
        test_data = features.iloc[split_idx:].copy()

        if len(train_data) < 10 or len(test_data) < 1:
            st.error("Pembagian data menghasilkan train/test yang tidak memadai.")
            st.stop()

        X_train = train_data[FEATURE_COLS].to_numpy()
        y_train = train_data["Y_load_current"].to_numpy()
        X_test = test_data[FEATURE_COLS].to_numpy()
        y_test = test_data["Y_load_current"].to_numpy()

        model = SugenoFIS(
            mf_name=selected_mf,
            width_scale=width_scale,
            ridge_alpha=ridge_alpha,
            q_low=q_low,
            q_high=q_high,
        ).fit(X_train, y_train)

        test_pred = np.array([model.explain_one(a, b)[0] for a, b in X_test])
        metrics = calculate_metrics(y_test, test_pred)
        baseline_metrics = calculate_metrics(y_test, X_test[:, 0])

        st.session_state.model_bundle = {
            "model": model,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "train_data": train_data,
            "test_data": test_data,
            "test_pred": test_pred,
            "config": config,
        }
        st.session_state.last_config = config

bundle = st.session_state.model_bundle
if bundle is None:
    st.stop()

model = bundle["model"]
train_data = bundle["train_data"]
test_data = bundle["test_data"]

# Ringkasan dan evaluasi
st.subheader("Ringkasan model")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jenis MF", selected_mf)
c2.metric("Jumlah aturan", "9")
c3.metric("Data latih", f"{len(train_data):,}")
c4.metric("Data uji", f"{len(test_data):,}")

st.subheader("Evaluasi pada data uji")
metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.markdown("**Sugeno FIS**")
    st.dataframe(
        pd.DataFrame([bundle["metrics"]]).round(4),
        use_container_width=True,
        hide_index=True,
    )
with metric_col2:
    st.markdown("**Baseline persistence (prediksi = X1)**")
    st.dataframe(
        pd.DataFrame([bundle["baseline_metrics"]]).round(4),
        use_container_width=True,
        hide_index=True,
    )

# Grafik perbandingan
st.subheader("Aktual vs prediksi pada data uji")
chart_data = pd.DataFrame({
    "Aktual": test_data["Y_load_current"].to_numpy(),
    "Prediksi Sugeno": bundle["test_pred"],
    "Baseline persistence": test_data[FEATURE_COLS[0]].to_numpy(),
}, index=test_data[TIMESTAMP_COL])
st.line_chart(chart_data.head(500))

# Interaksi input pengguna
st.divider()
st.subheader("3. Masukkan nilai untuk inferensi fuzzy")
st.write(
    "Masukkan dua nilai beban listrik. X1 adalah beban satu interval sebelumnya; "
    "X2 adalah beban pada jam yang sama di hari sebelumnya."
)

x1_min = float(train_data[FEATURE_COLS[0]].min())
x1_max = float(train_data[FEATURE_COLS[0]].max())
x2_min = float(train_data[FEATURE_COLS[1]].min())
x2_max = float(train_data[FEATURE_COLS[1]].max())

left, right = st.columns(2)
with left:
    input_x1 = st.number_input(
        "X1 — Load (t-1 interval)",
        min_value=float(min(0, x1_min)),
        max_value=float(max(x1_max * 1.5, x1_min + 1)),
        value=float(train_data[FEATURE_COLS[0]].median()),
        format="%.3f",
    )
with right:
    input_x2 = st.number_input(
        "X2 — Load (t-24 jam)",
        min_value=float(min(0, x2_min)),
        max_value=float(max(x2_max * 1.5, x2_min + 1)),
        value=float(train_data[FEATURE_COLS[1]].median()),
        format="%.3f",
    )

if st.button("Jalankan inferensi", type="primary"):
    prediction, membership_table, rules_table = model.explain_one(input_x1, input_x2)
    st.session_state.inference_result = {
        "prediction": prediction,
        "membership_table": membership_table,
        "rules_table": rules_table,
        "x1": input_x1,
        "x2": input_x2,
    }

if "inference_result" in st.session_state:
    result = st.session_state.inference_result
    st.markdown("### Hasil keputusan / output model")
    st.metric("Prediksi beban listrik (Y)", f"{result['prediction']:,.3f}")
    st.caption(
        f"Input: X1 = {result['x1']:,.3f} | X2 = {result['x2']:,.3f}"
    )

    st.markdown("### Tahap 1 — Derajat keanggotaan input")
    st.dataframe(
        result["membership_table"].style.format({
            "X1 — lag 1 interval": "{:.4f}",
            "X2 — lag 24 jam": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tahap 2 — Aktivasi aturan dan konsekuen Sugeno")
    st.caption(
        "Operator AND menggunakan perkalian. Kekuatan aktivasi dinormalisasi, "
        "lalu setiap aturan menghasilkan output lokal fᵢ = pᵢX1 + qᵢX2 + rᵢ."
    )
    st.dataframe(
        result["rules_table"].style.format({
            "Kekuatan aktivasi": "{:.5f}",
            "Output lokal aturan": "{:,.3f}",
            "Kontribusi tertimbang": "{:,.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Tahap 3 — Perhitungan output akhir")
    st.latex(r"\\hat{Y}=\\sum_{i=1}^{9}\\bar{w}_i f_i")
    st.write(
        "Output akhir adalah jumlah kontribusi tertimbang dari sembilan aturan. "
        "Pada Sugeno, agregasi akhir menggunakan weighted average; tidak memakai "
        "defuzzifikasi centroid seperti pada Mamdani."
    )

# Parameter membership yang sedang dipakai
with st.expander("Lihat parameter membership hasil estimasi dari data latih"):
    param_df = pd.DataFrame({
        "Himpunan": LINGUISTIC_LABELS,
        "X1": [model.params_x1[1], model.params_x1[2], model.params_x1[3]],
        "X2": [model.params_x2[1], model.params_x2[2], model.params_x2[3]],
    })
    st.write("Titik kuantil/representatif yang digunakan sebagai parameter MF:")
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    st.write("Rentang X1 (min, q-low, median, q-high, max):", model.params_x1)
    st.write("Rentang X2 (min, q-low, median, q-high, max):", model.params_x2)

st.caption(
    "Catatan: model dilatih dari data yang dipilih pada aplikasi. "
    "Parameter MF dihitung dari bagian train saja. Data uji kronologis digunakan "
    "untuk menampilkan metrik evaluasi, bukan untuk melatih konsekuen."
)
