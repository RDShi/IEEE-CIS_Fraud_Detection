import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

train_df = pd.read_pickle('/kaggle/input/ieee-test-data/trn.pkl')
test_df = pd.read_pickle('/kaggle/input/ieee-test-data/tst.pkl')

features_columns = ['V258_mean_payment_behavior', 'V258', 'D3_mean_payment_behavior', 'C14', 'C1', 'C13', 'V258_mean_uid', 'V257', 'R_emaildomain', 'TransactionAmt_log_mean_payment_behavior', 'V294', 'TransactionAmt_log', 'D3_std_payment_behavior', 'C10', 'C11', 'C8', 'TransactionAmt_log_std_payment_behavior', 'V201', 'D5_mean_payment_behavior', 'TransactionAmt_log_mean_uid', 'V317', 'D10_std_payment_behavior', 'V258_std_card_address', 'addr1_count_full', 'D3_mean_uid', 'V258_mean_user_group', 'D5_mean_uid', 'D3_std_uid', 'C2', 'D2', 'D7_mean_bank_type', 'D7_std_bank_type', 'V258_max_payment_behavior', 'P_emaildomain_count_full', 'D15_std_payment_behavior', 'TransactionAmt_log_std_uid', 'V258_mean_billing_region', 'V258_max_uid', 'D4_minus_dt_std_uid', 'D3_std_registration_date', 'D15_mean_uid', 'card1', 'D7_mean_activation_date', 'addr1', 'C12', 'V258_mean_bank_type', 'V258_std_billing_region', 'product_type_count_full', 'C6', 'C4', 'TransactionAmt_log_mean_bank_type', 'D3_mean_activation_date', 'V258_mean_card_address', 'D15_mean_payment_behavior', 'D15_std_uid', 'P_emaildomain', 'D4_mean_uid', 'V246', 'V70', 'D5_std_uid', 'V258_max_card_address', 'D5_std_registration_date', 'payment_behavior_count_full', 'D10_mean_payment_behavior', 'TransactionAmt_log_mean_registration_date', 'D10_minus_dt_std_uid', 'V258_std_user_group', 'D12_minus_dt_mean_bank_type', 'D15_minus_dt_count_full', 'D13_minus_dt_mean_bank_type', 'D10_minus_dt_std_payment_behavior', 'D14_minus_dt_std_bank_type', 'D7_mean_registration_date', 'card1_count_full', 'D10_std_uid', 'D8', 'D15_minus_dt_std_uid', 'TransactionAmt_log_std_bank_type', 'card2', 'D7_std_activation_date', 'D12_minus_dt_std_bank_type', 'V189', 'D4_std_uid', 'D13_minus_dt_std_bank_type', 'TransactionAmt_log_mean_activation_date', 'D14_std_registration_date', 'D3_std_bank_type', 'payment_behavior_nunique_card1', 'D3_mean_registration_date', 'D14_minus_dt_mean_bank_type', 'D14_minus_dt_std_registration_date', 'D3_std_activation_date', 'TransactionAmt_log_std_registration_date', 'D13_minus_dt_std_activation_date', 'D10_minus_dt_std_bank_type', 'D5_std_payment_behavior', 'D4_minus_dt_std_bank_type', 'D15_minus_dt_std_bank_type', 'V258_mean_registration_date', 'D7_std_registration_date', 'D14_minus_dt_std_activation_date', 'TransactionAmt_log_std_activation_date', 'D4_minus_dt_std_registration_date', 'D4_minus_dt_count_full', 'D5_mean_registration_date', 'TransactionAmt_decimal', 'id_20', 'D13_mean_registration_date', 'D3_mean_bank_type', 'D13_std_registration_date', 'D10_mean_uid', 'R_emaildomain_count_full', 'V258_std_bank_type', 'D12_std_activation_date', 'V187', 'D13_mean_activation_date', 'V258_std_registration_date', 'D5_mean_activation_date', 'D4_mean_payment_behavior', 'D12_minus_dt_std_activation_date', 'D5_mean_bank_type', 'D10_nunique_uid', 'id_02', 'uid_count_full', 'D13_minus_dt_std_registration_date', 'D10_minus_dt_count_full', 'D15_minus_dt_std_payment_behavior', 'M4', 'V45', 'D12_std_registration_date', 'id_31_count_full', 'D5_std_bank_type', 'D4_std_registration_date', 'D14_mean_registration_date', 'id_30', 'D12_minus_dt_std_registration_date', 'D14_mean_activation_date', 'D10_nunique_payment_behavior', 'D12_mean_registration_date', 'pca_V_2', 'D4_std_payment_behavior', 'V244', 'D1', 'D15_minus_dt_std_registration_date', 'V258_std_uid', 'D12_mean_activation_date', 'D15_minus_dt_mean_bank_type', 'long_V_0', 'D4_minus_dt_mean_bank_type', 'D5_std_activation_date', 'D14_std_activation_date', 'DeviceInfo_count_full', 'V258_std_activation_date', 'D15_std_registration_date', 'D15', 'D15_nunique_uid', 'D4_minus_dt_std_payment_behavior', 'D13_std_activation_date', 'D10_mean_registration_date', 'card2_count_full', 'product_type_nunique_uid', 'D15_minus_dt_std_activation_date', 'D4_mean_activation_date', 'D10_std_registration_date', 'id_01', 'V87', 'D15_mean_registration_date', 'M5', 'V313', 'D10_minus_dt_std_registration_date', 'D3_nunique_registration_date', 'D4_mean_registration_date', 'V258_max_billing_region', 'D10_std_activation_date', 'V258_mean_activation_date', 'V91', 'D15_std_activation_date', 'D15_nunique_payment_behavior', 'D4_minus_dt_std_activation_date', 'V258_max_bank_type', 'nulls1', 'D10_minus_dt_mean_bank_type', 'D3_nunique_payment_behavior', 'D4_nunique_uid', 'dist1', 'C7', 'D4_std_activation_date', 'D10_minus_dt_std_activation_date', 'V258_std_payment_behavior', 'D3_nunique_uid', 'card6', 'D13_nunique_registration_date', 'D10_nunique_registration_date', 'screen_height', 'D4', 'V258_max_registration_date', 'payment_behavior_nunique_card2', 'C5', 'D15_mean_activation_date', 'V323', 'D5_nunique_uid', 'D14_nunique_registration_date', 'id_19', 'D10_nunique_activation_date', 'D10_mean_activation_date', 'D14_minus_dt_count_full', 'V149', 'D14_minus_dt_nunique_registration_date', 'D13_minus_dt_nunique_activation_date', 'D3_nunique_activation_date', 'D12_minus_dt_nunique_registration_date', 'D4_nunique_registration_date', 'D7_nunique_bank_type', 'D12_mean_payment_behavior', 'D7_nunique_registration_date', 'V258_max_activation_date', 'D13_minus_dt_nunique_registration_date', 'registration_date_count_full', 'D5_nunique_registration_date', 'D4_nunique_activation_date', 'D4_nunique_payment_behavior', 'V53', 'D7_mean_uid', 'D12_nunique_registration_date', 'id_05', 'V156', 'V225', 'V312', 'D3_nunique_bank_type', 'D10_minus_dt_nunique_activation_date', 'activation_date_count_full', 'D15_nunique_registration_date', 'D4_minus_dt_nunique_uid', 'ProductCD_count_full', 'D10_minus_dt_nunique_registration_date', 'D15_nunique_activation_date', 'D14_minus_dt_nunique_activation_date', 'D4_minus_dt_nunique_activation_date', 'V258_max_user_group', 'C9', 'D14_nunique_activation_date', 'D5_nunique_activation_date', 'V62', 'device_name_count_full', 'V266', 'D15_minus_dt_nunique_registration_date', 'D4_minus_dt_nunique_payment_behavior', 'D13_nunique_activation_date', 'D13_minus_dt_count_full', 'V281', 'D12_minus_dt_nunique_bank_type', 'M6', 'D10', 'D15_minus_dt_nunique_uid', 'D5_nunique_payment_behavior', 'card6_count_full', 'D4_minus_dt_nunique_registration_date', 'D10_minus_dt_nunique_uid', 'id_06', 'V324', 'screen_width', 'D3', 'D4_minus_dt_nunique_bank_type', 'V44', 'D14_std_uid', 'card5', 'short_V_3', 'D13_minus_dt_nunique_bank_type', 'D14_mean_uid', 'D12_nunique_activation_date', 'D14_minus_dt_std_uid', 'D14_minus_dt_nunique_bank_type', 'D15_minus_dt_nunique_activation_date', 'D5_nunique_bank_type', 'D12_minus_dt_nunique_activation_date', 'D13_minus_dt_std_uid', 'D7_nunique_activation_date', 'id_33_count_full', 'D7_mean_payment_behavior', 'long_V_11', 'id_13', 'D15_minus_dt_nunique_payment_behavior', 'D14_mean_payment_behavior', 'V283', 'V209', 'D12_minus_dt_count_full', 'V61', 'D10_minus_dt_nunique_bank_type', 'D12_mean_uid', 'V314', 'short_V_14', 'V143', 'browser_type_count_full', 'D12_std_uid', 'id_17', 'V83', 'V165', 'D7_std_uid', 'dist2', 'D13_mean_uid', 'V130', 'V308', 'D12', 'D6', 'bank_type_count_full', 'M3', 'D10_minus_dt_nunique_payment_behavior', 'V310', 'short_V_8', 'D9', 'short_V_6', 'D15_minus_dt_nunique_bank_type', 'id_14', 'V30', 'D12_minus_dt_std_uid', 'V54', 'V90', 'V69', 'V200', 'D14', 'id_09', 'id_03', 'V86', 'D14_std_payment_behavior', 'ProductCD', 'short_V_5', 'short_V_4', 'card3', 'long_V_6', 'D5', 'DeviceType', 'D13_std_uid', 'D13', 'V76', 'long_V_2', 'V307', 'card5_count_full', 'M2', 'D13_mean_payment_behavior', 'D13_std_payment_behavior', 'D13_minus_dt_std_payment_behavior', 'id_18', 'D14_minus_dt_std_payment_behavior', 'pca_V_8', 'M9', 'card3_count_full', 'V12', 'V243', 'V82', 'pca_V_9', 'TransactionAmt_check', 'id_32', 'DeviceType_count_full', 'card4_count_full', 'short_V_9', 'short_V_7', 'long_V_5', 'D7_std_payment_behavior', 'D14_nunique_payment_behavior', 'payment_behavior_nunique_card5', 'V285', 'short_V_1', 'long_V_4', 'long_V_13', 'long_V_8', 'pca_V_0', 'D12_minus_dt_std_payment_behavior', 'V67', 'short_V_10', 'long_V_10', 'D12_std_payment_behavior', 'pca_V_13', 'long_V_9', 'V280', 'long_V_12', 'pca_V_11', 'D7_nunique_uid', 'D13_minus_dt_nunique_uid', 'M8', 'D14_minus_dt_nunique_uid', 'short_V_0', 'short_V_2', 'D12_minus_dt_nunique_uid', 'V296', 'D13_nunique_uid', 'pca_V_1', 'payment_behavior_nunique_card3', 'D12_nunique_uid', 'long_V_14', 'D14_nunique_uid', 'pca_V_14', 'C3', 'M7', 'id_38', 'id_04', 'long_V_1', 'pca_V_10', 'D14_minus_dt_nunique_payment_behavior', 'id_11', 'short_V_11', 'id_15', 'D7_nunique_payment_behavior', 'D7', 'pca_V_12', 'id_34', 'id_36', 'D13_nunique_payment_behavior', 'D13_minus_dt_nunique_payment_behavior', 'id_07', 'V65', 'id_21', 'V339', 'addr2_count_full', 'id_37', 'id_12', 'id_08', 'id_25', 'D12_minus_dt_nunique_payment_behavior', 'id_10', 'addr2', 'id_16', 'short_V_12', 'D12_nunique_payment_behavior', 'id_23', 'id_28', 'is_holiday', 'id_26', 'V41', 'id_35', 'id_24', 'M1', 'id_29', 'V1', 'V14', 'card4']

from catboost import CatBoostClassifier

cat_cols=['DeviceType', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'P_emaildomain', 'R_emaildomain',
          'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_12', 'id_15', 'id_16', 'id_23',
          'id_28', 'id_29', 'id_30', 'id_35', 'id_36', 'id_37', 'id_38']
for col in cat_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
random_state = 1337
oof_test = np.zeros((test_df.shape[0], 5))

for i in range(5):
    
    train_index = train_df[~(train_df['cv_id']==i)].index
    valid_index = train_df[(train_df['cv_id']==i)].index
    X_tr = train_df.loc[train_index, :]
    X_val = train_df.loc[valid_index, :]
    
    y_tr = X_tr['isFraud'].values
    X_tr = X_tr[features_columns]
#     X_tr[fillna_num] = X_tr[fillna_num].fillna(-999) 
    
    y_val = X_val['isFraud'].values
    X_val = X_val[features_columns]
#     X_val[fillna_num] = X_val[fillna_num].fillna(-999) 
    
    categorical_features_indices = np.where(~((X_tr.dtypes == np.float)+(X_tr.dtypes == np.int)))[0]
    print('\ny_tr distribution: {}'.format(Counter(y_tr)))
    
    
    print('training CatBoost:')
    model = CatBoostClassifier(iterations=10000,
                               learning_rate=0.07,
                               depth=15,
                               thread_count=-1,
                               loss_function="Logloss",
                               custom_metric=['Logloss', 'AUC'],
                               eval_metric='AUC',
                               task_type='GPU',
                               random_state=random_state,
                               use_best_model=True,
                               early_stopping_rounds=200,
#                                od_wait=300,
                               verbose=100)
    
    model.fit(X_tr, 
              y_tr, 
              cat_features=cat_cols,
              eval_set=(X_val, y_val), 
#               plot=True
             )

    X_test = test_df[features_columns]
#     X_test[fillna_num] = X_test[fillna_num].fillna(-999) 
    test_pred = model.predict_proba(X_test)[:,1]
    oof_test[:, i] = test_pred

test_df['isFraud'] = oof_test.mean(axis=1)
test_df[['TransactionID','isFraud']].to_csv('submission_cat.csv', index=False)


import xgboost as xgb

not_cat_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
train_df[not_cat_features] = train_df[not_cat_features].astype(float)
test_df[not_cat_features] = test_df[not_cat_features].astype(float)

cat_features = ['DeviceType', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 
                'P_emaildomain', 'R_emaildomain', 'ProductCD', 'id_12', 'id_15', 'id_16', 'id_23',
                'id_28', 'id_29', 'id_30', 'id_35', 'id_36', 'id_37', 'id_38']
train_df[cat_features] = train_df[cat_features].astype(int)
test_df[cat_features] = test_df[cat_features].astype(int)

fillna_num = ['V1', 'V12', 'V130', 'V14', 'V143', 'V149', 'V156', 'V165', 'V187', 'V189', 'V200', 'V201', 'V209', 
              'V225', 'V243', 'V244', 'V246', 'V257', 'V258', 'V266', 'V280', 'V281', 'V283', 'V285', 'V294', 'V296', 
              'V30', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V317', 'V323', 'V324', 'V339', 'V41', 'V44',
              'V45', 'V53', 'V54', 'V61', 'V62', 'V65', 'V67', 'V69', 'V70', 'V76', 'V82', 'V83', 'V86', 'V87', 'V90',
              'V91', 'pca_V_0', 'long_V_0', 'short_V_0', 'long_V_1', 'short_V_1', 'pca_V_1', 'long_V_2', 'short_V_2',
              'pca_V_2', 'short_V_3', 'long_V_4', 'short_V_4', 'long_V_5', 'short_V_5', 'long_V_6', 'short_V_6', 
              'short_V_7', 'long_V_8', 'short_V_8', 'pca_V_8', 'long_V_9', 'short_V_9', 'pca_V_9', 'long_V_10', 
              'short_V_10', 'pca_V_10', 'long_V_11', 'short_V_11', 'pca_V_11', 'long_V_12', 'short_V_12', 'pca_V_12',
              'long_V_13', 'pca_V_13', 'long_V_14', 'short_V_14', 'pca_V_14']

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta':0.01,
    'max_depth': 20,
    'seed': 1337,
    'subsample': 1.0,
    'colsample_bytree': 0.5,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
}
early_stop = 200
num_rounds = 10000
verbose_eval = 100

oof_test = np.zeros((test_df.shape[0], 5))

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
    
    d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
    d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    print('\ntraining XGB:')
    model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                      early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=xgb_params)

    X_test = test_df[features_columns]
    X_test[fillna_num] = X_test[fillna_num].fillna(-999) 
    test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

    oof_test[:, i] = test_pred
    
test_df['isFraud'] = oof_test.mean(axis=1)
test_df[['TransactionID','isFraud']].to_csv('submission_xgb.csv', index=False)