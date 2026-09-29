# =============================================================================
# DECISION TREE - GINI INDEX (Multi-way Split, Tinggi = 2)
# Studi Kasus : Prediksi Arah Harga Saham BBRI
# Versi       : VS Code (lokal) - tanpa dependensi eksternal selain pandas
# =============================================================================
#
# CARA MENJALANKAN DI VS CODE:
#   Terminal → python decision_tree_gini_vscode.py
#
# FILE YANG DIBUTUHKAN (harus ada di folder yang sama):
#   - training_bbri_colab.csv
#   - test_bbri_colab.csv
#   - forward_bbri_colab.csv
# =============================================================================

import os
import sys
import io
import pandas as pd
from collections import Counter

# Fix encoding Windows terminal (cp1252 -> utf-8)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 0. PASTIKAN WORKING DIRECTORY = FOLDER SCRIPT INI
#    (sehingga CSV selalu ditemukan meski dijalankan dari mana saja)
# ─────────────────────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# 1. MUAT DATA
# ─────────────────────────────────────────────────────────────────────────────
train_df   = pd.read_csv("training_bbri_colab.csv")
test_df    = pd.read_csv("test_bbri_colab.csv")
forward_df = pd.read_csv("forward_bbri_colab.csv")

ATTRS = ["Arah_HariIni", "Volume_Kategori", "Rentang_Volatilitas", "Gap_Pembukaan"]
LABEL = "Label_Keputusan"

# Ubah ke list-of-tuples agar cepat diproses tanpa pandas
train   = list(train_df[ATTRS + [LABEL]].itertuples(index=False, name=None))
test    = list(test_df[ATTRS + [LABEL]].itertuples(index=False, name=None))

# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNGSI GINI INDEX
# ─────────────────────────────────────────────────────────────────────────────
def gini(counter: Counter) -> float:
    """Hitung Gini impurity dari distribusi kelas."""
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counter.values())


def weighted_gini(subset: list, attr_idx: int):
    """
    Hitung Weighted Gini untuk atribut tertentu.
    Mengelompokkan data berdasarkan nilai atribut, lalu
    menjumlah Gini tiap grup secara tertimbang.
    """
    groups = {}
    for row in subset:
        groups.setdefault(row[attr_idx], []).append(row)

    n = len(subset)
    wg, detail = 0.0, {}
    for val, rows in groups.items():
        cnt = Counter(r[-1] for r in rows)
        g   = gini(cnt)
        wg += (len(rows) / n) * g
        detail[val] = {"distribusi": dict(cnt), "gini": round(g, 4)}

    return round(wg, 4), detail


def best_split(subset: list, exclude: list = None) -> str:
    """
    Pilih atribut terbaik (weighted Gini terkecil).
    Atribut dalam `exclude` tidak dipertimbangkan (cegah duplikasi).
    """
    exclude = exclude or []
    parent_cnt = Counter(r[-1] for r in subset)

    print(f"n={len(subset)} | distribusi={dict(parent_cnt)} | - decision_tree_gini_vscode.py:85"
          f"Gini_parent={round(gini(parent_cnt), 4)}")

    results = {}
    for i, attr in enumerate(ATTRS):
        if attr in exclude:
            continue
        wg, detail = weighted_gini(subset, i)
        results[attr] = wg
        print(f"[{attr}] weighted_Gini = {wg} - decision_tree_gini_vscode.py:94")
        for val, info in detail.items():
            print(f"nilai='{val}' → {info} - decision_tree_gini_vscode.py:96")

    best_attr = min(results, key=results.get)
    print(f"✔ Split terpilih : {best_attr}  (weighted_Gini = {results[best_attr]})\n - decision_tree_gini_vscode.py:99")
    return best_attr

# ─────────────────────────────────────────────────────────────────────────────
# 3. INDUKSI DECISION TREE (TINGGI = 2, MULTI-WAY SPLIT)
# ─────────────────────────────────────────────────────────────────────────────
print("= - decision_tree_gini_vscode.py:105" * 65)
print("INDUKSI DECISION TREE  GINI INDEX - decision_tree_gini_vscode.py:106")
print("Studi kasus : Prediksi Arah Harga Saham BBRI - decision_tree_gini_vscode.py:107")
print("= - decision_tree_gini_vscode.py:108" * 65)

print("\n╔══════════════════════════════╗ - decision_tree_gini_vscode.py:110")
print("║        ROOT  (Level 1)       ║ - decision_tree_gini_vscode.py:111")
print("╚══════════════════════════════╝ - decision_tree_gini_vscode.py:112")
root_attr = best_split(train)
root_idx  = ATTRS.index(root_attr)

# Kelompokkan data training berdasarkan nilai atribut root
groups_l1 = {}
for row in train:
    groups_l1.setdefault(row[root_idx], []).append(row)

tree        = {}    # kunci: (val_level1, attr_level2, val_level2) → label
leaf_stats  = []    # untuk hitung training error

print("╔══════════════════════════════╗ - decision_tree_gini_vscode.py:124")
print("║      CABANG  (Level 2)       ║ - decision_tree_gini_vscode.py:125")
print("╚══════════════════════════════╝ - decision_tree_gini_vscode.py:126")

for val1, subset in sorted(groups_l1.items()):
    print(f"\n  ┌─ {root_attr} = '{val1}'  (n={len(subset)}) - decision_tree_gini_vscode.py:129")
    cnt = Counter(r[-1] for r in subset)

    # Kondisi daun langsung (data murni atau terlalu sedikit)
    if len(cnt) == 1 or len(subset) < 2:
        label = cnt.most_common(1)[0][0]
        err   = len(subset) - cnt.most_common(1)[0][1]
        print(f"└── LEAF → '{label}'  (murni/sangat sedikit) - decision_tree_gini_vscode.py:136"
              f"error={err}/{len(subset)}")
        tree[(val1, None, None)] = label
        leaf_stats.append((err, len(subset)))
        continue

    # Cari atribut split Level 2
    attr2 = best_split(subset, exclude=[root_attr])
    idx2  = ATTRS.index(attr2)

    # Bagi lagi berdasarkan atribut Level 2
    groups_l2 = {}
    for row in subset:
        groups_l2.setdefault(row[idx2], []).append(row)

    for val2, sub2 in sorted(groups_l2.items()):
        cnt2  = Counter(r[-1] for r in sub2)
        label = cnt2.most_common(1)[0][0]
        err   = len(sub2) - cnt2.most_common(1)[0][1]
        print(f"│     {attr2}='{val2}' → LEAF='{label}' - decision_tree_gini_vscode.py:155"
              f"dist={dict(cnt2)}  error={err}/{len(sub2)}")
        tree[(val1, attr2, val2)] = label
        leaf_stats.append((err, len(sub2)))

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING ERROR & ESTIMASI GENERALIZATION ERROR
# ─────────────────────────────────────────────────────────────────────────────
total_err = sum(e for e, _ in leaf_stats)
total_n   = sum(n for _, n in leaf_stats)
n_leaf    = len(leaf_stats)

OMEGA = 0.5   # penalti per daun (pessimistic pruning)
optimistic  = total_err / total_n
pessimistic = (total_err + OMEGA * n_leaf) / total_n

print("\n - decision_tree_gini_vscode.py:171" + "=" * 65)
print("TRAINING ERROR - decision_tree_gini_vscode.py:172")
print("= - decision_tree_gini_vscode.py:173" * 65)
print(f"Jumlah daun (leaf)  : {n_leaf} - decision_tree_gini_vscode.py:174")
print(f"Jumlah data training: {total_n} - decision_tree_gini_vscode.py:175")
print(f"Total error         : {total_err} - decision_tree_gini_vscode.py:176")
print(f"Training error      : {total_err}/{total_n} = {optimistic*100:.2f}% - decision_tree_gini_vscode.py:177")
print(f"Optimistic estimate : {optimistic*100:.2f}% - decision_tree_gini_vscode.py:178")
print(f"Pessimistic estimate (Ω={OMEGA}): {pessimistic*100:.2f}% - decision_tree_gini_vscode.py:179")
print("MDL : disesuaikan skema encoding dari dosen masingmasing. - decision_tree_gini_vscode.py:180")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
majority_class = Counter(r[-1] for r in train).most_common(1)[0][0]

def predict(row: tuple) -> str:
    """
    Telusuri pohon: cocokkan nilai Level 1, lalu Level 2.
    Jika kombinasi belum pernah dilihat saat training → kembalikan kelas mayoritas.
    """
    v1 = row[root_idx]
    for (val1, attr2, val2), label in tree.items():
        if val1 != v1:
            continue
        if attr2 is None:          # daun langsung di level 1
            return label
        idx2 = ATTRS.index(attr2)
        if row[idx2] == val2:
            return label
    return majority_class          # fallback


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUASI PADA TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n - decision_tree_gini_vscode.py:207" + "=" * 65)
print(f"EVALUASI TEST SET  ({len(test)} baris) - decision_tree_gini_vscode.py:208")
print("= - decision_tree_gini_vscode.py:209" * 65)

test_errors = 0
print(f"\n  {'No':<5} {'Aktual':<10} {'Prediksi':<10} {'Status'} - decision_tree_gini_vscode.py:212")
print(f"{''*4} {''*9} {''*9} {''*8} - decision_tree_gini_vscode.py:213")
for i, row in enumerate(test):
    pred   = predict(row)
    actual = row[-1]
    status = "✔" if pred == actual else "✘ SALAH"
    print(f"{i+1:<5} {actual:<10} {pred:<10} {status} - decision_tree_gini_vscode.py:218")
    if pred != actual:
        test_errors += 1

test_accuracy = (len(test) - test_errors) / len(test)
test_error_rate = test_errors / len(test)

# ─────────────────────────────────────────────────────────────────────────────
# 7. RINGKASAN AKHIR
# ─────────────────────────────────────────────────────────────────────────────
print("\n - decision_tree_gini_vscode.py:228" + "=" * 65)
print("RINGKASAN HASIL - decision_tree_gini_vscode.py:229")
print("= - decision_tree_gini_vscode.py:230" * 65)
print(f"Atribut ROOT (Level 1)   : {root_attr} - decision_tree_gini_vscode.py:231")
print(f"Total data training      : {total_n} baris - decision_tree_gini_vscode.py:232")
print(f"Total data test          : {len(test)} baris - decision_tree_gini_vscode.py:233")
print(f"Training error           : {optimistic*100:.2f}% - decision_tree_gini_vscode.py:234")
print(f"Optimistic estimate      : {optimistic*100:.2f}% - decision_tree_gini_vscode.py:235")
print(f"Pessimistic estimate     : {pessimistic*100:.2f}% - decision_tree_gini_vscode.py:236")
print(f"Test error aktual        : {test_error_rate*100:.2f}% - decision_tree_gini_vscode.py:237")
print(f"Test accuracy aktual     : {test_accuracy*100:.2f}% - decision_tree_gini_vscode.py:238")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PREDIKSI DATA FORWARD (SESI BERIKUTNYA)
# ─────────────────────────────────────────────────────────────────────────────
fwd_row  = list(forward_df[ATTRS].itertuples(index=False, name=None))[0]
pred_fwd = predict(fwd_row)

print("\n - decision_tree_gini_vscode.py:246" + "=" * 65)
print("PREDIKSI SESI BERIKUTNYA - decision_tree_gini_vscode.py:247")
print("= - decision_tree_gini_vscode.py:248" * 65)
for attr, val in zip(ATTRS, fwd_row):
    print(f"{attr:<28}: {val} - decision_tree_gini_vscode.py:250")
print(f"\n  ► Prediksi keputusan  : {pred_fwd} - decision_tree_gini_vscode.py:251")
print()
print("⚠  PERINGATAN: Hasil ini adalah OUTPUT MODEL EDUKATIF. - decision_tree_gini_vscode.py:253")
print("Bukan rekomendasi investasi. Selalu lakukan analisis - decision_tree_gini_vscode.py:254")
print("fundamental & teknikal secara mandiri sebelum berinvestasi. - decision_tree_gini_vscode.py:255")
print("= - decision_tree_gini_vscode.py:256" * 65)
