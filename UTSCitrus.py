import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # type: ignore

# pertama load dataset terlebih dahulu
df = pd.read_csv("citrus.csv")

# kedua pisahkan fitur
X = df.drop('name', axis=1)
y = df['name']

# ketiga Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# keempat split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# kelima buat model naive bayes
model = GaussianNB()
model.fit(X_train, y_train)

# meminta prediksi 
y_pred = model.predict(X_test)

# 7. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le.classes_)

print("======= clusterfikasi jeruk dan anggur menggunakan metode naive bayen =======")
print("Akurasi: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
