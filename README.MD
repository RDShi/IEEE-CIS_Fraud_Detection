本题的主要思路是利用信息定位出 uid，然后进行各种 agg。

主要的 magic 是 D1-dt，再利用：
- V95, V97, V96 are previous transaction counts in day, week, month
- V126, V128, V127 are cumulative amount of transactions made by user in previous day, week, month.
- V306, V308, V307 are cumulative amount of transactions made by user in previous day, week, month.

定位出更精细的uid。

Public Leaderboard: 42nd

Private Leaderboard: 38th

跟 top rank team 相比，缺失了更精细的 uid 的定位。

```
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def cal_sub_id(df_base, eps = 1e-2):
    
    df = df_base.copy()
    df.reset_index(level=0, inplace=True)
    df['sub_id'] = -1

    sub_id = 1
    start_i = 0
    
    while start_i < len(df):
        if df.loc[start_i, 'sub_id'] > 0:
            start_i += 1
            continue
        df.loc[start_i, 'sub_id'] = sub_id
        i1 = start_i
        for i2 in range(i1+1, len(df)):
            if df.loc[i2, 'sub_id'] > 0:
                continue
            cond = (df.index < i2) & (df['sub_id'] == sub_id)
            sum_month = df.loc[(df['DT'] >= df.loc[i2, 'DT'] - timedelta(days=30) ) & cond, "TransactionAmt"].sum()
            sum_week  = df.loc[(df['DT'] >= df.loc[i2, 'DT'] - timedelta(days=7)  ) & cond, "TransactionAmt"].sum()
            sum_day =   df.loc[(df['DT'] >= df.loc[i2, 'DT'] - timedelta(hours=24)) & cond, "TransactionAmt"].sum()
            
            if ((abs(df.loc[i2, 'D1'] - df.loc[i1, 'D1'] - df.loc[i2, 'D3']) < 2 or np.isnan(df.loc[i2, 'D3'])) and \
                ((df.loc[i2, 'D2'] >= df.loc[i1, 'D2']) or np.isnan(df.loc[i2, 'D2']) or np.isnan(df.loc[i1, 'D2'])) and \
                ((df.loc[i2, 'V307'] <= eps + df.loc[i1, 'V307'] + df.loc[i1, 'TransactionAmt'] and df.loc[i2, 'V307'] >= sum_month - eps) or np.isnan(df.loc[i2, 'V307']) or np.isnan(df.loc[i1, 'V307']) ) and \
                ((df.loc[i2, 'V308'] <= eps + df.loc[i1, 'V308'] + df.loc[i1, 'TransactionAmt'] and df.loc[i2, 'V308'] >= sum_week - eps) or np.isnan(df.loc[i2, 'V308']) or np.isnan(df.loc[i1, 'V308']) ) and \
                ((df.loc[i2, 'V306'] <= eps + df.loc[i1, 'V306'] + df.loc[i1, 'TransactionAmt'] and df.loc[i2, 'V306'] >= sum_day - eps) or np.isnan(df.loc[i2, 'V306']) or np.isnan(df.loc[i1, 'V306']) )):
                i1 = i2
                df.loc[i1, 'sub_id'] = sub_id
        sub_id += 1

    return pd.Series(df['sub_id'].values, index=df_base.index)


df = pd.merge(df, applyParallel(df.groupby(['uid']), cal_sub_id).rename('sub_id'), on=['TransactionID'], how='inner')
```
