# LAPORAN PROYEK MESIN LEARNING

### Nama : Raufan Ahsan Haerizal

### Nim : 211351119

### Kelas : Pagi A

## Domain Proyek

Proyek ini dapat digunakan untuk Mencari Berat Dari Cereal Membangun sistem otomatis untuk mengukur dan mengisi berat cereal ke dalam kemasan. Ini dapat mencakup penggunaan berat yang tepat untuk masing-masing kemasan atau porsi.

## Business Understanding

Rating dari cereal dalam konteks bisnis adalah penilaian atau evaluasi atas kualitas, popularitas, dan performa produk sereal. Penilaian ini bisa dilakukan oleh berbagai pihak, seperti konsumen, produsen, atau ahli industri makanan.

### Problem Statements

Problem statement dalam konteks rating sereal adalah pernyataan yang mengidentifikasi masalah atau tantangan yang perlu diatasi terkait dengan penilaian atau rating produk sereal.

### Goals

Mencari Rating Dari cereal baik untuk kesehatan apa tidak

### Solution statements

 panduan atau rencana tindakan yang akan membantu dalam mengatasi masalah atau mencapai tujuan yang terkait dengan produk sereal atau bisnis yang berkaitan dengan sereal.

 ## Data Understanding

 Dataset yang saya gunakan berasal dari Kaggle yang berisi 80 Cereals. Dataset ini merupakan sekumpulan data yang dikumpulkan dari website 80 Cereal. Dataset ini mengandung lebih dari 10 columns setelah dilakukan data cleaning..
 <br>
 [80 Cereals] (https://www.kaggle.com/datasets/crawford/80-cereals)

### Variabel-variabel pada 80 Cereals adalah sebagai berikut:
- calories = banyaknya kalori dalam cereal
- protein =  banyaknya protein dalam cereal
- fat = beberapa fat di dalam cereal
- sodium = banyaknya sodium/natrium dalam cereal
- fiber = banyaknya fiber atau serat di dalam cereal
- carbo = beberapa Carbo di dalam cereal
- sugars = beberapa sugar di dalam cereal
- weight = beberapa berat/weight dari cereal

## Data Preparation

### Data Collection
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama dataset 80 Cereals, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

### Data Discovery And Profiling
Untuk bagian ini, kita akan menggunakan teknik EDA. <br>
Pertama kita mengimport semua library yang dibutuhkan,
``` bash
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
Mari lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable, dan melihat 5 data paling atas dari datasetsnya
``` bash
df = pd.read_csv('cereal.csv')
df.head()
```
Untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan property info,
``` bash
df.info()
```
``` bash
print("Dataset Columns and rows:", df.shape)
print("Dataset size:", df.size)
```
``` bash
df.describe()
```
Mari kita lanjut dengan data exploration kita,
``` bash
plt.figure(figsize=(8,6))
plt.title("Overall Cereals Data Distribution")
sns.histplot(data = cereals)
```
mengecek atau memastikan kalo yang null itu true atau false,
``` bash
cereals.isnull().any()
```
lalu check mengunakan yand di bawah dan bakal keluar hasilnya,
``` bash
cereals.duplicated().any()
```
## Modeling
Fungsi ini akan membuat gambar atau model grafik
``` bash
def plot_histogram(column_data, column_name):
    plt.figure(figsize=(5, 3))
    plt.title(f"Distribution of {column_name}")
    sns.histplot(column_data, kde=True) 
    plt.show()
```
![Alt text](output10.png)

lalu ini memunculkan grafik manufacturer
``` bash
plot_histogram(cereals['mfr'], 'Manufacturer')
```
![Alt text](output9.png)

ini memunculkan grafik type 
``` bash
plot_histogram(cereals['type'], 'Cold or Hot Types of Cereal')
```
![Alt text](output8.png)

ini memunculkan grafik calories
``` bash
plot_histogram(cereals['calories'], 'Calories')
```
![Alt text](output7.png)

ini memunculkan grafik protein
``` bash
plot_histogram(cereals['protein'], 'Protein')
```
![Alt text](output6.png)

ini memunculkan grafik fat
``` bash
plot_histogram(cereals['fat'], 'Fat')
```
![Alt text](output5.png)

ini memunculkan grafik sodium
``` bash
plot_histogram(cereals['sodium'], 'Sodium')
```
![Alt text](output4.png)

ini memunculkan grafik potass
``` bash
plot_histogram(cereals['potass'], 'potass')
```
![Alt text](output3.png)

ini memunculkan grafik rating
``` bash
plot_histogram(cereals['rating'], 'Consumer Rating')
```
![Alt text](output1.png)

ini memunculkan grafik sugar
``` bash
plot_histogram(cereals['sugars'], 'Sugar')
```
![Alt text](output.png)
Kode ini menunjukkan bagaimana Anda mengambil beberapa fitur (features)
``` bash
features = ['calories','protein','fat','sodium','fiber','carbo','sugars','weight']
x = df[features]
y = df['rating']
x.shape, y.shape
```
Kode ini menunjukkan model selection (features)
``` bash
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=100)
y_test.shape
```
lalu ini untuk memindahkan linier model ke linear regression
``` bash
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
ini code untik memunculkan akurasi dari regresi linier
``` bash
score = lr.score(X_test, y_test)
print('akurasi model regresi linier = ', score)
```
lalu masukan data sesuai yang kalian masukan ke features dan prediction
``` bash
input_data = np.array([[120,3,5,15,2,8,8,1]])

prediction = lr.predict(input_data)
print('prediksi rating di dalam creal :', prediction)
```
wow, berhasil!!, sekarang modelnya sudah selesai, mari kita export sebagai sav agar nanti bisa kita gunakan pada project web streamlit kita.
``` bash
import pickle

filename = 'prediksi_creal.sav'
pickle.dump(lr,open(filename,'wb'))
```

## Evaluation
Disini saya menggunakan linear regression buat memunculkan akurasi.
- regresi linear untuk memodelkan hubungan antara atribut-atribut tersebut dan bobot (weight) suatu produk makanan. 

- Setelah itu saya menerapkannya dalam kode menggunakan fungsi linier regression, seperti berikut :
``` bash 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)

score = lr.score(X_test, y_test)
print('akurasi model regresi linier = ', score)
```
``` bash
Y = a + bX
```

dan hasil yang saya dapatkan adalah 0.9934779670066929 atau 99.3%, itu berarti model ini memiliki keseimbangan yang baik antara presisi dan recall. Karena kita mencari patokan harga untuk membeli Apartment maka model yang presisi sangat dibutuhkan agar kemungkinan terjadinya kesalahan semakin sedikit.

## Deployment

[My Estimation App] [https://opannn-raufan.streamlit.app/]

![Image1](sa.png)
