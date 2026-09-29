# =============================================================
# Decision Tree Classifier (sklearn) - dataset BBRI riil
# criterion='gini', max_depth=2
# =============================================================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ---------- 1. Load data ----------
train_df = pd.read_csv("training_bbri_colab.csv")
test_df  = pd.read_csv("test_bbri_colab.csv")
forward_df = pd.read_csv("forward_bbri_colab.csv")

FEATURES = ["Arah_HariIni", "Volume_Kategori", "Rentang_Volatilitas", "Gap_Pembukaan"]
LABEL = "Label_Keputusan"

# ---------- 2. Encode atribut kategorikal jadi angka (sklearn butuh numerik) ----------
encoders = {}
X_train = pd.DataFrame()
X_test = pd.DataFrame()
X_forward = pd.DataFrame()

for col in FEATURES:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col], forward_df[col]]))
    encoders[col] = le
    X_train[col] = le.transform(train_df[col])
    X_test[col] = le.transform(test_df[col])
    X_forward[col] = le.transform(forward_df[col])

y_train = train_df[LABEL]
y_test = test_df[LABEL]

# ---------- 3. Training ----------
dt_model = DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=42)
dt_model.fit(X_train, y_train)

# ---------- 4. Evaluasi pada data training ----------
y_pred_train = dt_model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
train_error = 1 - train_acc

# ---------- 5. Evaluasi pada data test ----------
y_pred_test = dt_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
test_error = 1 - test_acc

# ---------- 6. Prediksi ke depan (belum diketahui jawabannya) ----------
pred_forward = dt_model.predict(X_forward)[0]

# =============================================================
# RINGKASAN OUTPUT
# =============================================================
print("=" * 60)
print("RINGKASAN TRAINING DECISION TREE (sklearn)".center(60))
print("=" * 60)

print(f"\n[1] Parameter model: criterion=gini, max_depth=2")
print(f"    Fitur yang dipakai: {FEATURES}")

print(f"\n[2] TRAINING ERROR")
print(f"    Akurasi training : {train_acc*100:.2f}%")
print(f"    Error training   : {train_error*100:.2f}%")

print(f"\n[3] TEST ERROR (data nyata, {len(test_df)} baris, ground truth diketahui)")
print(f"    Akurasi test : {test_acc*100:.2f}%")
print(f"    Error test   : {test_error*100:.2f}%")
print(f"    Confusion matrix:\n{confusion_matrix(y_test, y_pred_test)}")

print(f"\n[4] PREDIKSI KE DEPAN (jawaban sungguhan belum ada)")
print(f"    Fitur : {forward_df[FEATURES].iloc[0].to_dict()}")
print(f"    Hasil : {pred_forward}")

print("\n" + "=" * 60)

# ---------- 7. Visualisasi pohon (gambar) ----------
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
plot_tree(
    dt_model,
    feature_names=FEATURES,
    class_names=dt_model.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
    impurity=True,
    ax=ax
)
plt.title("Decision Tree - Prediksi Arah Harga BBRI (criterion=gini, max_depth=2)", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree_sklearn.png", dpi=150, bbox_inches="tight")
print("Gambar pohon disimpan sebagai: decision_tree_sklearn.png")
plt.show()
