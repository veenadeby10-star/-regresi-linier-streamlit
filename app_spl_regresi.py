## Aplikasi Penyelesaian SPL (Regresi Linear - Normal Equation) dengan Streamlit
## Jalankan dengan:  streamlit run app_spl_regresi.py
## Dependensi: pip install streamlit pandas numpy openpyxl xlrd
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Fungsi-fungsi bantu
# ---------------------------------------------------------------------------

def baca_file(file):
    """Membaca file CSV atau Excel menjadi DataFrame."""
    isi = file.getvalue()                      # selalu ambil seluruh byte (aman saat rerun)
    nama = file.name.lower()

    if nama.endswith(".csv"):
        for enc in ("utf-8-sig", "latin-1"):   # utf-8-sig menangani BOM dari Excel
            try:
                df = pd.read_csv(BytesIO(isi), sep=None, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Encoding file CSV tidak dikenali.")
    else:
        try:
            df = pd.read_excel(BytesIO(isi))
        except ImportError as e:
            raise ImportError(
                f"{e}. Install dengan: pip install openpyxl (untuk .xlsx) "
                "atau pip install xlrd (untuk .xls)"
            )

    df.columns = [str(c).strip() for c in df.columns]   # rapikan spasi pada nama kolom
    return df


def bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep):
    """Membentuk matriks desain X dan vektor target y dari DataFrame."""
    data = df[kolom_fitur + [kolom_target]].apply(pd.to_numeric, errors="coerce")
    if data.isnull().any().any():
        raise ValueError("Terdapat sel kosong/non-angka pada kolom yang dipilih.")

    X = data[kolom_fitur].to_numpy(dtype=float)
    y = data[kolom_target].to_numpy(dtype=float)

    nama_w = [f"w{i+1}" for i in range(len(kolom_fitur))]
    if pakai_intersep:
        X = np.column_stack([np.ones(len(X)), X])   # tambah kolom 1 di kiri
        nama_w = ["w0"] + nama_w
    return X, y, nama_w


def selesaikan_normal_equation(X, y):
    """Menyelesaikan (X^T X) w = X^T y dengan solver."""
    n, p = X.shape
    if n < p:
        raise ValueError(
            f"Jumlah sampel ({n}) kurang dari jumlah parameter ({p}); "
            "SPL tidak punya solusi tunggal."
        )
    if np.linalg.matrix_rank(X) < p:
        raise ValueError(
            "Matriks XᵀX singular (tidak punya invers). Kemungkinan ada fitur yang "
            "saling bergantung linear (mis. kolom konstan atau duplikat)."
        )
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    return XtX, Xty, w


def susun_persamaan(target, nama_w, w, fitur, intersep):
    """Menyusun string persamaan model dengan tanda +/- yang rapi."""
    nama_fitur = ([None] + fitur) if intersep else fitur
    hasil = ""
    for i, (nilai, f) in enumerate(zip(w, nama_fitur)):
        angka = f"{abs(nilai):.4f}" + ("" if f is None else f"·{f}")
        if i == 0:
            hasil += ("-" if nilai < 0 else "") + angka
        else:
            hasil += (" - " if nilai < 0 else " + ") + angka
    return f"{target} = {hasil}"


# ---------------------------------------------------------------------------
# Tampilan utama
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SPL Regresi Linear", layout="wide")
st.title("Penyelesaian SPL dengan Streamlit")
st.caption("Regresi linear berganda melalui normal equation: (XᵀX) w = Xᵀy")

file = st.file_uploader("Unggah file data (CSV / Excel)", type=["csv", "xlsx", "xls"])

if file is not None:
    try:
        df = baca_file(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.subheader("1. Data yang diunggah")
    st.dataframe(df)

    # ----- Pilih kolom -----
    kolom_numerik = df.select_dtypes(include="number").columns.tolist()
    if len(kolom_numerik) < 2:
        st.error("Data harus memiliki minimal 2 kolom numerik (fitur dan target).")
        st.stop()

    st.subheader("2. Pilih variabel")
    col1, col2 = st.columns(2)
    with col1:
        kolom_target = st.selectbox(
            "Kolom target (y)", kolom_numerik, index=len(kolom_numerik) - 1
        )
    with col2:
        abaikan = ("sampel", "no", "id", "nomor")
        opsi_fitur = [k for k in kolom_numerik if k != kolom_target]
        default_fitur = [k for k in opsi_fitur if k.lower() not in abaikan]
        kolom_fitur = st.multiselect("Kolom fitur (x)", opsi_fitur, default=default_fitur)
    pakai_intersep = st.checkbox("Tambahkan intersep (w0)", value=True)

    # ----- Tombol hitung -----
    if st.button("Selesaikan", type="primary"):
        try:
            if not kolom_fitur:
                raise ValueError("Pilih minimal satu kolom fitur.")
            X, y, nama_w = bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep)
            XtX, Xty, w = selesaikan_normal_equation(X, y)
            st.session_state["hasil"] = dict(
                file=file.name, fitur=list(kolom_fitur), target=kolom_target,
                X=X, y=y, nama_w=nama_w, XtX=XtX, Xty=Xty, w=w,
                intersep=pakai_intersep,
            )
        except Exception as e:
            st.session_state.pop("hasil", None)
            st.error(f"Terjadi kesalahan: {e}")

    # ----- Tampilkan hasil -----
    h = st.session_state.get("hasil")
    if h and h["file"] == file.name:
        if (h["fitur"] != list(kolom_fitur) or h["target"] != kolom_target
                or h["intersep"] != pakai_intersep):
            st.warning("Pilihan variabel berubah sejak perhitungan terakhir. "
                       "Klik **Selesaikan** lagi untuk memperbarui hasil.")

        X, y, nama_w = h["X"], h["y"], h["nama_w"]
        w = h["w"] + 0.0   # menghindari tampilan -0.0

        st.subheader("3. Langkah perhitungan")
        kolom_X = (["1"] + h["fitur"]) if h["intersep"] else h["fitur"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriks desain X**")
            st.dataframe(pd.DataFrame(X, columns=kolom_X))
        with c2:
            st.markdown("**Vektor y**")
            st.dataframe(pd.DataFrame(y, columns=[h["target"]]))

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**XᵀX**")
            st.dataframe(pd.DataFrame(h["XtX"], index=nama_w, columns=nama_w))
        with c4:
            st.markdown("**Xᵀy**")
            st.dataframe(pd.DataFrame(h["Xty"], index=nama_w, columns=["Xᵀy"]))

        st.subheader("4. Solusi w")
        hasil_w = pd.DataFrame({"parameter": nama_w, "nilai": np.round(w, 4)})
        st.dataframe(hasil_w, hide_index=True)

        st.info("Model: " + susun_persamaan(h["target"], nama_w, w, h["fitur"], h["intersep"]))

        # Evaluasi
        y_hat = X @ w
        galat = y - y_hat
        sse = float(np.sum(galat ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst > 0 else float("nan")
        m1, m2 = st.columns(2)
        m1.metric("RMSE", f"{np.sqrt(np.mean(galat ** 2)):.4f}")
        m2.metric("R²", f"{r2:.4f}")

        st.markdown("**Perbandingan nilai aktual dan prediksi**")
        df_pred = pd.DataFrame({"aktual": y, "prediksi": np.round(y_hat, 4),
                                "galat": np.round(galat, 4)})
        st.dataframe(df_pred)

        st.download_button(
            "Unduh hasil prediksi (CSV)",
            df_pred.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi.csv", mime="text/csv",
        )

        # Prediksi data baru
        st.subheader("5. Prediksi data baru")
        nilai_baru = []
        kolom_in = st.columns(len(h["fitur"]))
        for kol, nama in zip(kolom_in, h["fitur"]):
            rata = pd.to_numeric(df[nama], errors="coerce").mean()
            rata = 0.0 if pd.isna(rata) else float(rata)
            nilai_baru.append(kol.number_input(nama, value=rata, key=f"baru_{nama}"))
        x_baru = np.array(([1.0] if h["intersep"] else []) + nilai_baru)
        st.success(f"Prediksi {h['target']} = {float(x_baru @ w):.2f}")
else:
    st.info("Silakan unggah file CSV/Excel terlebih dahulu. Contoh: data_rumah.csv")