import numpy as np
import pandas as pd
import gc
from collections import Counter
import lightgbm as lgb


def lgb_prediction(train_df,
                   test_df, 
                   params, 
                   features_columns, 
                   fillna_num, 
                   target,
                   return_oof=False,
                   cv_num=5, 
                   verbose_eval=100):
    
    oof_train = np.zeros((train_df.shape[0]))
    oof_test = np.zeros((test_df.shape[0], cv_num))
    models = []
    
    try:
        for i in range(cv_num):

            train_index = train_df[~(train_df['cv_id']==i)].index
            valid_index = train_df[(train_df['cv_id']==i)].index
            X_tr = train_df.loc[train_index, :]
            X_val = train_df.loc[valid_index, :]

            y_tr = X_tr[target].values
            X_tr = X_tr[features_columns]
            X_tr[fillna_num] = X_tr[fillna_num].fillna(-999) 

            y_val = X_val[target].values
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
            models.append(model)

            if return_oof:
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                oof_train[valid_index] = val_pred

            X_test = test_df[features_columns]
            X_test[fillna_num] = X_test[fillna_num].fillna(-999) 
            test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            oof_test[:, i] = test_pred
        
    except KeyboardInterrupt:
        print("break down")
    
    if return_oof:
        return models, oof_test, oof_train
    return models, oof_test

