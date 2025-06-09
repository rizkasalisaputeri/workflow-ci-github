import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

# Mengaktifkan MLflow Autologging untuk Scikit-learn
mlflow.sklearn.autolog()

# Memuat dataset yang sudah diproses
df = pd.read_csv("heart_processed.csv")

# Memisahkan fitur (X) dan target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Memulai MLflow run
with mlflow.start_run(run_name="Basic_RandomForest_Autolog_Full_Metrics"):
    
    # Inisialisasi model
    model = RandomForestClassifier(random_state=42)
    
    # Melatih model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Akurasi  : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
