import pandas as pd
from collections import Counter

# ---------- 1. Load data ----------
train_df = pd.read_csv("training_bbri_colab.csv")
test_df  = pd.read_csv("test_bbri_colab.csv")
forward_df = pd.read_csv("forward_bbri_colab.csv")

ATTRS = ["Arah_HariIni", "Volume_Kategori", "Rentang_Volatilitas", "Gap_Pembukaan"]
LABEL = "Label_Keputusan"

train = list(train_df[ATTRS + [LABEL]].itertuples(index=False, name=None))
test  = list(test_df[ATTRS + [LABEL]].itertuples(index=False, name=None))

# ---------- 2. Fungsi GINI ----------
def gini(counter):
    n = sum(counter.values())
    if n == 0:
        return 0
    return 1 - sum((c / n) ** 2 for c in counter.values())

def weighted_gini(subset, attr_idx):
    groups = {}
    for row in subset:
        groups.setdefault(row[attr_idx], []).append(row)
    n = len(subset)
    wg, detail = 0, {}
    for val, rows in groups.items():
        cnt = Counter(r[-1] for r in rows)
        g = gini(cnt)
        wg += (len(rows) / n) * g
        detail[val] = (dict(cnt), round(g, 4))
    return round(wg, 4), detail

def best_split(subset, exclude=None):
    exclude = exclude or []
    parent_cnt = Counter(r[-1] for r in subset)
    print(f"n={len(subset)}  distribusi={dict(parent_cnt)}  Gini_parent={round(gini(parent_cnt),4)} - decision_tree_gini_colab.py:38")
    results = {}
    for i, a in enumerate(ATTRS):
        if a in exclude:
            continue
        wg, detail = weighted_gini(subset, i)
        results[a] = wg
        print(f"{a}: weighted_Gini={wg} | {detail} - decision_tree_gini_colab.py:45")
    best_attr = min(results, key=results.get)
    print(f">> Split terpilih: {best_attr} (Gini={results[best_attr]}) - decision_tree_gini_colab.py:47")
    return best_attr

# ---------- 3. Induksi tree (tinggi = 2, multi-way split) ----------
print("=== ROOT (Level 1) === - decision_tree_gini_colab.py:51")
root_attr = best_split(train)
root_idx = ATTRS.index(root_attr)

groups_l1 = {}
for row in train:
    groups_l1.setdefault(row[root_idx], []).append(row)

tree = {}          # (val_level1, attr_level2, val_level2) -> label
leaf_stats = []     # untuk hitung training error

for val1, subset in groups_l1.items():
    print(f"\n Cabang {root_attr} = {val1} (n={len(subset)}) - decision_tree_gini_colab.py:63")
    cnt = Counter(r[-1] for r in subset)
    if len(cnt) == 1 or len(subset) < 2:
        label = cnt.most_common(1)[0][0]
        err = len(subset) - cnt.most_common(1)[0][1]
        print(f"LEAF > {label} (murni/terlalu kecil), error={err}/{len(subset)} - decision_tree_gini_colab.py:68")
        tree[(val1, None, None)] = label
        leaf_stats.append((err, len(subset)))
    else:
        attr2 = best_split(subset, exclude=[root_attr])
        idx2 = ATTRS.index(attr2)
        groups_l2 = {}
        for row in subset:
            groups_l2.setdefault(row[idx2], []).append(row)
        for val2, sub2 in groups_l2.items():
            cnt2 = Counter(r[-1] for r in sub2)
            label = cnt2.most_common(1)[0][0]
            err = len(sub2) - cnt2.most_common(1)[0][1]
            print(f"Leaf ({attr2}={val2}): dist={dict(cnt2)} > label={label}, error={err}/{len(sub2)} - decision_tree_gini_colab.py:81")
            tree[(val1, attr2, val2)] = label
            leaf_stats.append((err, len(sub2)))

# ---------- 4. Training error & estimasi generalization error ----------
total_err = sum(e for e, n in leaf_stats)
total_n = sum(n for e, n in leaf_stats)
n_leaf = len(leaf_stats)

print(f"\n=== TRAINING ERROR === - decision_tree_gini_colab.py:90")
print(f"Jumlah leaf = {n_leaf}, N = {total_n}, Error = {total_err} - decision_tree_gini_colab.py:91")
print(f"Training error = {total_err}/{total_n} = {total_err/total_n*100:.2f}% - decision_tree_gini_colab.py:92")

optimistic = total_err / total_n
omega = 0.5   # penalti per node, bisa disesuaikan sesuai materi dosen
pessimistic = (total_err + omega * n_leaf) / total_n
print(f"Optimistic approach   : {optimistic*100:.2f}% - decision_tree_gini_colab.py:97")
print(f"Pessimistic approach (Ω={omega}): {pessimistic*100:.2f}% - decision_tree_gini_colab.py:98")
print("MDL: sesuaikan dengan skema encoding yang diajarkan dosen Anda. - decision_tree_gini_colab.py:99")

# ---------- 5. Fungsi prediksi ----------
def predict(row):
    v1 = row[root_idx]
    for (val1, attr2, val2), label in tree.items():
        if val1 == v1:
            if attr2 is None:
                return label
            idx2 = ATTRS.index(attr2)
            if row[idx2] == val2:
                return label
    # fallback: kelas mayoritas keseluruhan (jika kombinasi tak terlihat saat training)
    return Counter(r[-1] for r in train).most_common(1)[0][0]

# ---------- 6. Evaluasi ke data test (ground truth diketahui) ----------
print(f"\n=== EVALUASI PADA TEST SET ({len(test)} baris) === - decision_tree_gini_colab.py:115")
test_err = 0
for row in test:
    pred = predict(row)
    actual = row[-1]
    if pred != actual:
        test_err += 1
test_rate = test_err / len(test)
print(f"Test error = {test_err}/{len(test)} = {test_rate*100:.2f}% - decision_tree_gini_colab.py:123")

print(f"\n=== RINGKASAN === - decision_tree_gini_colab.py:125")
print(f"Training error       : {optimistic*100:.2f}% - decision_tree_gini_colab.py:126")
print(f"Optimistic estimate  : {optimistic*100:.2f}% - decision_tree_gini_colab.py:127")
print(f"Pessimistic estimate : {pessimistic*100:.2f}% - decision_tree_gini_colab.py:128")
print(f"Test error aktual    : {test_rate*100:.2f}% - decision_tree_gini_colab.py:129")

# ---------- 7. Prediksi ke depan (belum diketahui jawabannya) ----------
fwd = list(forward_df[ATTRS].itertuples(index=False, name=None))[0]
pred_fwd = predict(fwd)
print(f"\n=== PREDIKSI SESI BERIKUTNYA === - decision_tree_gini_colab.py:134")
print(f"Fitur: {dict(zip(ATTRS, fwd))} - decision_tree_gini_colab.py:135")
print(f"Prediksi: {pred_fwd}  (catatan: bukan rekomendasi investasi, murni hasil model edukatif) - decision_tree_gini_colab.py:136")
