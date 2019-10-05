import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import datetime
import gc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import tqdm
from scipy.stats import ks_2samp

from utils.EDA import *
from utils.training_model import *
from utils.preprocessing import *

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

random_state = 1337

warnings.simplefilter('ignore')
sns.set()

df = pd.read_feather('../features/basic_info_data.ftr')
print('df.shape: ',df.shape)

df_V = pd.read_feather("../features/df_about_V.ftr")
V_cols = []
for col in df_V:
    V_cols.append(col)
    df[col] = df_V[col]
print('df.shape: ',df.shape)


%%time
# uids
uids = ['bank_type', 'billing_region', 'user_group', 'uid', 'payment_behavior', 
        'registration_date', 'activation_date', 'card_address']

df['registration_date'] = df['D1'] - df['day_id']
df['activation_date'] = df['D2'].fillna(0) - df['day_id']

df['bank_type'] = df['card2'].astype(str)+'_'+ df['card3'].astype(str)+'_'+ df['card5'].astype(str)

df['card_address'] = df['card1'].astype(str)+'_'+df['bank_type']
df['billing_region'] = df['bank_type'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
df['user_group'] = df['billing_region'].astype(str)+'_'+df[
    'P_emaildomain'].astype(str)+'_'+df['R_emaildomain'].astype(str)

df['uid'] = (df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)
             +'_'+df['card4'].astype(str)+'_'+df['card5'].astype(str)+'_'+df['card6'].astype(str)
             +'_'+df['registration_date'].astype(str))

df['payment_behavior'] = (df['uid']+'_'+df['R_emaildomain'].astype(str)
                          +'_'+df['P_emaildomain'].astype(str)
                          +'_'+df['addr1'].astype(str)
                          +'_'+df['addr2'].astype(str)
                          +'_'+df['activation_date'].astype(str))


D_t_list = ['D4', 'D10', 'D12', 'D13', 'D14', 'D15'] #, 'D11'
D_dt_list = ['D3', 'D5', 'D7']

for Dx in D_t_list:
    df[Dx+'_minus_dt'] = df[Dx] - df['day_id']

agg_map = {}
for uid in uids:
    agg_map[uid+'$V258'] = ['mean', 'std', 'max']
    if uid not in ['billing_region', 'card_address', 'user_group']:
        for col in D_dt_list:
            agg_map[uid+'$'+col] = ['mean', 'std', 'nunique']
        
        for col in D_t_list:
            if uid in ['uid', 'registration_date', 'activation_date', 'payment_behavior']:
                agg_map[uid+'$'+col] = ['mean', 'std', 'nunique']
                agg_map[uid+'$'+col+'_minus_dt'] = ['std', 'nunique']
            else:
                agg_map[uid+'$'+col+'_minus_dt'] = ['mean', 'std', 'nunique']
                
        agg_map[uid+'$TransactionAmt_log'] = ['mean', 'std']

for col in ['card1', 'card2', 'card3', 'card5']:
    agg_map[col+'$payment_behavior'] = ['nunique']

agg_map['payment_behavior$product_type'] = ['nunique']

# df = batch_aggregation(agg_map, df)
print('df.shape: ',df.shape)



df = batch_aggregation(agg_map, df)
print('df.shape: ',df.shape)


# freq enconding features
for col in uids+['D4_minus_dt', 'D10_minus_dt', 'D12_minus_dt', 'D13_minus_dt', 'D14_minus_dt', 'D15_minus_dt']:
    print(col)
    df = count_encoding(col, df)
    
print('df.shape: ',df.shape)


## Label Encoder
self_encoding_feat = ['product_type', 'device_name', 'id_31', 'browser_type', 'id_33', 'DeviceInfo']
for col in list(df):
    if (col not in uids+self_encoding_feat) and df[col].dtype=='O':
        print(col)
        df[col] = df[col].fillna('unseen_before_label')
        
        df[col] = df[col].astype(str)
        df[col] = df[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(df[col]))
        df[col] = le.transform(df[col])
        df[col] = df[col].astype('category')

print('df.shape: ',df.shape)

%%time
train_df = df[~df['isFraud'].isna()].reset_index(drop=True)
test_df = df[df['isFraud'].isna()].reset_index(drop=True)

if 'cv_id' in train_df.columns:
    train_df = train_df.drop('cv_id',axis=1)
cvid = pd.read_csv('cvid.csv')
train_df = pd.merge(train_df, cvid, on='TransactionID', how='left')

print('train_df.shape: ',train_df.shape)
print('test_df.shape: ',test_df.shape)
# train_df.shape:  (590540, x)
# test_df.shape:  (506691, x)


%%time
target = 'isFraud'

noise_features = ['TransactionID', 'TransactionDT', 'cv_id', 'TransactionAmt'] + [
    'month_id','week_id','day_id','hour_id', 'hour','day_in_month', 'dow'] +[
    'month_id_total','week_id_total','day_id_total','hour_id_total'] + [
#     'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
    'D11']

importance_V = ['V258', 'V257', 'V294', 'V317', 'V246', 'V201', 'V70', 'V187', 'V243', 'V308', 'V244', 
                'V283', 'V45', 'V62', 'V91', 'V200', 'V156', 'V313', 'V87', 'V312', 'V69', 'V44', 'V189',
                'V281', 'V30', 'V67', 'V225', 'V323', 'V149', 'V314', 'V54', 'V296', 'V53', 'V324', 'V130',
                'V83', 'V12', 'V143', 'V307', 'V90', 'V266', 'V61', 'V76', 'V280', 'V165', 'V209', 'V310', 
                'V86', 'V285', 'V82']
binary_V = ['V1', 'V14', 'V41', 'V65', 'V88', 'V107', 'V305']

useless_V = ['V107', 'V305', 'V88']
for Vx in ['V'+str(i) for i in range(1,339)]:
    if (Vx not in importance_V) and (Vx not in binary_V):
        useless_V.append(Vx)

fillna_num = V_cols+[]
useless_feat = ['id_22', 'id_27']
overfit_feat = []
overfit_feat = ['D15_minus_dt', 'D4_minus_dt', 'D10_minus_dt', 'D14_minus_dt', 'D12_minus_dt', 'D13_minus_dt',
                'user_group_count_full', 'card_address_count_full', 'billing_region_count_full',
                'uid_device_hash_nunique_in_card3', 'uid_device_hash_nunique_in_card5',
                'V258_mean_payment_behavior', 'V258_max_in_uid_device_hash',
                'D3_std_activation_date','D3_std_uid', 'D7_std_registration_date',
                'TransactionAmt_log_mean_uid', 'D15_mean_uid', 'D15_mean_in_uid_device_hash',
                'D7_mean_registration_date', 'D3_mean_registration_date', 'D4_mean_in_uid_device_hash']

rm_cols = ['isFraud'] + self_encoding_feat + noise_features + useless_feat + uids + overfit_feat + useless_V

features_columns = list(train_df)
for col in rm_cols:
    if col in features_columns:
        features_columns.remove(col)
    if col in fillna_num:
        fillna_num.remove(col)
        
        

verbose_eval = 100
random_state = 1337
params = {
    "pos_scale_weight": 10,
    "learning_rate":0.01,
    "num_leaves":2**9,
    "objective":'binary',
    "min_child_weight": 1,
#     "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 1e-3,
    "n_estimators":5000, 
    "bagging_fraction":1.0,
    "bagging_freq": 0, 
    "feature_fraction":0.5,
    "lambda_l1":1e-5,
    "lambda_l2":0.0,
    "max_bin": 255,
#     "min_data_in_bin": 3,
    "importance_type":"gain",
    "metric": ['auc', 'binary_logloss'],
    "n_jobs": -1,
    'early_stopping_rounds':100,
    'n_estimators': 10000,
    "random_state": random_state,
    'seed': random_state,
    'feature_fraction_seed': random_state,
    'bagging_seed': random_state,
    'drop_seed': random_state,
    'data_random_seed': random_state,
    'random_state': random_state,
    "first_metric_only": True,
    "save_binary": True
}

oof_train = np.zeros((train_df.shape[0]))
oof_test = np.zeros((test_df.shape[0], 5))

models = []

for i in range(5):
    
    train_index = train_df[~(train_df['cv_id']==i)].index
    valid_index = train_df[(train_df['cv_id']==i)].index
    X_tr = train_df.loc[train_index, :]
    X_val = train_df.loc[valid_index, :]
    
    y_tr = X_tr['isFraud'].values
    X_tr = X_tr[features_columns]
    X_tr[fillna_num] = X_tr[fillna_num].fillna(-999) 
    
    y_val = X_val['isFraud'].values
    X_val = X_val[features_columns]
    X_val[fillna_num] = X_val[fillna_num].fillna(-999) 
    
    print('\ny_tr distribution: {}'.format(Counter(y_tr)))
    print('y_val distribution: {}'.format(Counter(y_val)))
    print('X_tr shape: {}'.format(X_tr.shape))
    print('X_val shape: {}'.format(X_val.shape))
    
    d_train = lgb.Dataset(X_tr, label=y_tr)
    d_valid = lgb.Dataset(X_val, label=y_val)
    watchlist = [d_train, d_valid]
    
    print('\ntraining LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval)
    
#     val_pred = model.predict(X_val, num_iteration=model.best_iteration)
#     oof_train[valid_index] = val_pred
    
    X_test = test_df[features_columns]
    X_test[fillna_num] = X_test[fillna_num].fillna(-999) 
    test_pred = model.predict(X_test, 
                              num_iteration=model.best_iteration)
    oof_test[:, i] = test_pred
    models.append(model)


auc_list = []
logloss_list = []
for model in models:
    auc_list.append(model.best_score['valid_1']['auc'])
    logloss_list.append(model.best_score['valid_1']['binary_logloss'])
print("{:.6f}/{:.6f}, {:.6f}/{:.6f}, {:.6f}/{:.6f}, {:.6f}/{:.6f}, {:.6f}/{:.6f}".format(logloss_list[0],
                                                                                         auc_list[0],
                                                                                         logloss_list[1],
                                                                                         auc_list[1],
                                                                                         logloss_list[2],
                                                                                         auc_list[2],
                                                                                         logloss_list[3],
                                                                                         auc_list[3],
                                                                                         logloss_list[4],
                                                                                         auc_list[4]))
print(np.mean(auc_list))

prediction = pd.read_csv('../input/sample_submission.csv')
prediction['isFraud'] = oof_test.mean(axis=1)
prediction[['TransactionID','isFraud']].to_csv('submission_'+str(np.mean(auc_list))+'.csv', index=False)


model = models[0]
iteration = model.best_iteration
# iteration = 10
feature_import = list(zip(model.feature_name(),
                          model.feature_importance(importance_type='gain', iteration=iteration), 
                          model.feature_importance(importance_type='split', iteration=iteration)))
feature_import.sort(key=lambda x:x[1],reverse=True)
features = []

print("{:<10} {:<30} {:<20} {:<10}".format('rank', 'feature', 'gain', 'split'))
index = 1
for line in feature_import:
    print("{:<10} {:<30} {:<20} {:<10}".format(index, line[0], line[1], line[2]))
    index += 1
    features.append(line[0])


# for kernel
train_df_for_kernel = reduce_mem_usage(train_df[features_columns+['TransactionID', 'isFraud', 'cv_id']], verbose=True)
test_df_for_kernel = reduce_mem_usage(test_df[features_columns+['TransactionID']], verbose=True)
train_df_for_kernel.to_pickle('trn.pkl')
test_df_for_kernel.to_pickle('tst.pkl')



