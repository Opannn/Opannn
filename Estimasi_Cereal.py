import pickle
import streamlit as st

model = pickle.load(open('estimasi_creal.sav', 'rb'))

st.title('prediksi nilai Rating di dalam creal')

calories = st.number_input('Input calories dari Cereal')
protein = st.number_input('Input protein dari Cereal')
fat = st.number_input('Input fat dari Cereal')
sodium = st.number_input('Input sodium dari Cereal')
fiber = st.number_input('Input fiber dari Cereal ')
carbo = st.number_input('Input carbo dari Cereal')
sugars = st.number_input('Input sugar dari Cereal')
weight = st.number_input('Input weight dari Cereal')

predict = ''

if st.button('Prediksi rating'):
    predict = model.predict(
        [[calories,protein,fat,sodium,fiber,carbo,sugars,weight]]
    )
    st.write ('Hasil Dari Prediksi Rating : ', predict)
