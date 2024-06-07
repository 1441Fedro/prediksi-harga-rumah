import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model_regresi.joblib')

with st.sidebar:
    st.sidebar.title("Tentang Saya")

    st.sidebar.subheader("Nama",divider='blue')
    st.sidebar.write("Fedro Maulana Jatmika")

    st.sidebar.subheader("NPM",divider='blue')
    st.sidebar.write("50421503")

    st.sidebar.subheader("Kelas",divider='blue')
    st.sidebar.write("3IA26")

# membuat fungsi untuk prediksi
def regresi(kt,km,p,lt,lb):
    fitur = np.array([[kt,km,p,lt,lb]])
    prediksi = model.predict(fitur)
    return prediksi[0]

# Streamlit app
st.header("Prediksi Harga Rumah di Kota Bekasi",anchor=None,divider='blue')
st.write("Masukkan kriteria yang anda inginkan:")
kt = st.slider("Jumlah Kamar Tidur", 0, 10, 0)
km = st.slider("Jumlah Kamar Mandi", 0, 10, 0)
p = st.slider("Kapasitas Parkir Mobil", 0, 10, 0)
lt = st.slider("Luas Tanah", 0, 200, 0)
lb = st.slider("Luas Bangunan", 0, 200, 0)

if st.button("Predict"):
    prediksi = regresi(kt,km,p,lt,lb)
    rentang_bawah = prediksi
    rentang_atas1 = prediksi + 279049479
    rentang_atas2 = prediksi + 153810900
    st.write(f"Hasil prediksi memiliki rentang harga (rmse): Rp {rentang_bawah:.0f} - Rp {rentang_atas1:.0f}")
    st.write(f"Hasil prediksi memiliki rentang harga (mae): Rp {rentang_bawah:.0f} - Rp {rentang_atas2:.0f}")
