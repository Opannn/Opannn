import pickle
import streamlit as st

model = pickle.load(open('estimasi_creal.sav', 'rb'))

st.title('Estimasi nilai weight di dalam creal')

calories = st.number_input('Input calories')
protein = st.number_input('Input protein')
fat = st.number_input('Input fat')
sodium = st.number_input('Input sodium')
fiber = st.number_input('Input fiber')
carbo = st.number_input('Input carbo')
sugars = st.number_input('Input sugars')
rating = st.number_input('Input rating')

predict = ''

if st.button('Estimasi Weight'):
    predict = model.predict(
        [[calories,protein,fat,sodium,fiber,carbo,sugars,rating]]
    )
    st.write ('Estimasi Creal weight : ', predict)