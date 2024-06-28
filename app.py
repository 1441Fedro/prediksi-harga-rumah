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
kt = st.number_input("Jumlah Kamar Tidur", 0, 10, 0)
km = st.number_input("Jumlah Kamar Mandi", 0, 10, 0)
p = st.number_input("Kapasitas Parkir Mobil", 0, 10, 0)
lt = st.number_input("Luas Tanah", 0, 200, 0)
lb = st.number_input("Luas Bangunan", 0, 200, 0)

if st.button("Predict"):
    prediksi = regresi(kt,km,p,lt,lb)
    # harga prediksi rumah
    rentang_bawah = prediksi
    rentang_atas1 = prediksi + 144846019

    # mengubah ke int
    rentang_bawah_int = int(rentang_bawah)
    rentang_atas1_int = int(rentang_atas1)

    # menambah titik pemisah
    rentang_bawah_format = f"{rentang_bawah_int:,}".replace(',', '.')
    rentang_atas1_format = f"{rentang_atas1_int:,}".replace(',', '.')

    st.markdown("### Hasil prediksi memiliki rentang harga (RMSE): ")
    st.markdown(f"### :blue-background[Rp {rentang_bawah_format} - Rp {rentang_atas1_format}] ")
