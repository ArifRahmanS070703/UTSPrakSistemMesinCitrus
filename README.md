# UTSPrakSistemMesinCitrus
Saya Arif Rahman Sopian dengan NIM 1217050020. Disini saya mengerjakan suatu project UTS Praktikum Sistem Mesin dengan Mengklasifikasi Buah Jeruk Dan Anggur Menggunakan metode Naive Bayen. dengan Dosen Pengampu: Bapak Aldy Rialdy Atmadja, MT.

1. Mencantumkan Library
import pandas as pd # memuat dan mengelola data 
from sklearn.model_selection import train_test_split # data pelatihan dan pengujian
from sklearn.naive_bayes import GaussianNB # model Naive Bayes
from sklearn.preprocessing import LabelEncoder # label kategori menjadi angka 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # mengevaluasi kinerja model

2.Membaca Dataset
df = pd.read_csv("citrus.csv")
*P: Dataset citrus.csv dimuat ke dalam df

3. Memisahkan Label sama Firur
X = df.drop('name', axis=1)
y = df['name']
*P:X adalah fitur (data input)
*P:y adalah label (data target)

4. Encode Label
le = LabelEncoder()
y_encoded = le.fit_transform(y)
*P: Kolom name biasanya berupa string seperti "jeruk", "anggur".model ML cuman menerima angka..
LabelEncoder mengubah:
"jeruk" → 0
"anggur" → 1

6. Membagi Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

7. Melatih Model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

8. Memprediksi dan mengevaluasi
y_pred = model.predict(X_test)

9. Evaluasi Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le.classes_)

9.Melihatkan Hasil
print("======= clusterfikasi jeruk dan anggur menggunakan metode naive bayen =======")
print("Akurasi: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
 
