import os, pandas as pd
from collections import Counter
os.chdir(r'C:\Users\Administrator\Downloads\Tugas Data Mining')

train_df = pd.read_csv('training_bbri_colab.csv')
ATTRS = ['Arah_HariIni','Volume_Kategori','Rentang_Volatilitas','Gap_Pembukaan']
LABEL = 'Label_Keputusan'
train = list(train_df[ATTRS + [LABEL]].itertuples(index=False, name=None))

def gini(counter):
    n = sum(counter.values())
    if n == 0: return 0
    return 1 - sum((c/n)**2 for c in counter.values())

def weighted_gini(subset, attr_idx):
    groups = {}
    for row in subset:
        groups.setdefault(row[attr_idx], []).append(row)
    n = len(subset)
    wg = 0
    for val, rows in groups.items():
        cnt = Counter(r[-1] for r in rows)
        g = gini(cnt)
        wg += (len(rows)/n) * g
    return round(wg,4)

print('==================================================')
print('LANGKAH 1: DATA TRAINING')
print('==================================================')
cnt_all = Counter(r[-1] for r in train)
n = len(train)
print('Total data :', n, 'baris')
print('Kelas Beli :', cnt_all['Beli'], 'baris')
print('Kelas Tidak:', cnt_all['Tidak'], 'baris')
beli_prob  = cnt_all['Beli']/n
tidak_prob = cnt_all['Tidak']/n
gini_root  = round(1 - beli_prob**2 - tidak_prob**2, 4)
print('P(Beli)    :', cnt_all['Beli'], '/', n, '=', round(beli_prob,4))
print('P(Tidak)   :', cnt_all['Tidak'], '/', n, '=', round(tidak_prob,4))
print('Gini ROOT  : 1 - ('+str(round(beli_prob,4))+')^2 - ('+str(round(tidak_prob,4))+')^2 =', gini_root)

print()
print('==================================================')
print('LANGKAH 2: HITUNG WEIGHTED GINI TIAP ATRIBUT')
print('==================================================')

for i, attr in enumerate(ATTRS):
    groups = {}
    for row in train:
        groups.setdefault(row[i], []).append(row)

    wg = 0
    print()
    print('[' + attr + ']')
    for val in sorted(groups.keys()):
        rows = groups[val]
        cnt  = Counter(r[-1] for r in rows)
        g    = gini(cnt)
        wg  += (len(rows)/n) * g
        print('  nilai="' + str(val) + '" : n=' + str(len(rows)) +
              ', Beli=' + str(cnt.get('Beli',0)) +
              ', Tidak=' + str(cnt.get('Tidak',0)) +
              ', Gini=' + str(round(g,4)))
    print('  Weighted_Gini =', round(wg,4))

print()
print('==================================================')
print('LANGKAH 3: PILIH ATRIBUT SPLIT TERBAIK (ROOT)')
print('==================================================')
wg_results = {}
for i, attr in enumerate(ATTRS):
    wg_results[attr] = weighted_gini(train, i)
for attr, wg in sorted(wg_results.items(), key=lambda x: x[1]):
    mark = ' <-- TERPILIH (minimum)' if wg == min(wg_results.values()) else ''
    print(' ', attr.ljust(30), ':', wg, mark)

print()
print('==================================================')
print('LANGKAH 4: BAGI DATA BERDASARKAN ROOT SPLIT')
print('==================================================')
root_attr = 'Rentang_Volatilitas'
root_idx  = ATTRS.index(root_attr)
groups_l1 = {}
for row in train:
    groups_l1.setdefault(row[root_idx], []).append(row)

for val, subset in sorted(groups_l1.items()):
    cnt = Counter(r[-1] for r in subset)
    g   = round(gini(cnt), 4)
    print()
    print('Cabang', root_attr, '=', val, '| n =', len(subset))
    print('  Distribusi:', dict(cnt))
    print('  Gini cabang ini:', g)

print()
print('==================================================')
print('LANGKAH 5: SPLIT LEVEL 2 PER CABANG')
print('==================================================')
for val1, subset in sorted(groups_l1.items()):
    print()
    print('-- Cabang: Rentang_Volatilitas =', val1, '(n=' + str(len(subset)) + ')')
    results = {}
    for i, attr in enumerate(ATTRS):
        if attr == root_attr:
            continue
        results[attr] = weighted_gini(subset, i)
        print('   [' + attr + '] Weighted_Gini =', results[attr])
    best = min(results, key=results.get)
    print('   >> Split Level 2 terpilih:', best, '(Gini=' + str(results[best]) + ')')
