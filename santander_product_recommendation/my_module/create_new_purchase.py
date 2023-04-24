import pandas as pd
import numpy as np

contest_name = 'santander-product-recommendation'

# 訓練データの読み込み
trn = pd.read_csv('../dataset/train_ver2.csv')
# 商品変数をprodsにlist形式で保存
prods = trn.columns[24:].tolist()

# 日付を数字に変換する変数（2015-01-28:1, ... , 2016-06-28:18）
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    int_date = (Y-2015)*12+M
    return int_date

# 日付を数字に変換し、int_dateに保存
trn['int_date'] = trn['fecha_dato'].map(date_to_int).astype(np.int8)

# データをコピーし、int_dateの日付に１を加えてlagを生成。識別番号と日付以外のカラム名に_prevを追加
# lagデータは時系列データにおける過去データのこと
trn_lag = trn.copy()
trn_lag['int_date'] += 1
trn_lag.columns = [col+'_prev' if col not in ['ncodpers', 'int_date'] else col for col in trn.columns]

# 原本データとlagデータを識別番号と日付を基準として合わせる
# lagデータの日付は1だけ押されているので、前の月の商品情報が挿入される
df_trn = trn.merge(trn_lag, on=['ncodpers', 'int_date'], how='left')

# メモリ解放
del trn, trn_lag

# 前の月の商品情報が存在しない場合に備え、0に代替
for prod in prods:
    prev = prod+'_prev'
    df_trn[prev].fillna(0, inplace=True)

# 原本データで商品を保有しているか、-lagデータで商品を保有してるかを比較し、新規購買変数paddを求める
for prod in prods:
    padd = prod+'_add'
    prev = prod+'_prev'
    df_trn[padd] = ((df_trn[prod] == 1) & (df_trn[prev] == 0)).astype(np.int8)

# 新規購買変数だけを抽出し、labelsに保存します
add_cols = [prod + '_add' for prod in prods]
labels = df_trn[add_cols].copy()
labels.columns = prods
labels.to_csv('../dataset/labels.csv', index=False)

