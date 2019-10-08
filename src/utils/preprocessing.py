import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# aggregation features
def batch_aggregation(agg_map, df):
    """
    Using col2's agg by grouping col1 as new features
    agg_map: {col$col2: [aggtypes]}
    """
    for cols in agg_map:
        agg_types = agg_map[cols]
        col1, col2 = cols.split('$')
        print(col2+', '+col1+': '+str(agg_types))
        df_tmp = df[[col1,col2]].copy()
        
        if agg_types[0] != 'nunique':
            df_tmp[col2] = df_tmp[col2].astype('float')
        else:
            df_tmp[col2] = df_tmp[col2].astype('str')
            
        df_dict = df_tmp.groupby([col1])[col2].agg(agg_types)
        for agg_type in agg_types:
            df[col2+'_'+agg_type+'_in_'+col1] = df[col1].map(df_dict[agg_type])
    return df


# freq enconding features
def count_encoding(col, df):
    df[col+'_count_full'] = df[col].map(df[col].value_counts(dropna=True))
    return df


# Conversion of Category Characteristics
def encoding_obj(df, ec_feat = None, not_ec_feat = []):
    if ec_feat==None:
        ec_feat = list(df)
    
    for col in ec_feat:
        if (col not in not_ec_feat) and df[col].dtype=='O':
            print(col)
            df[col] = df[col].fillna('unseen_before_label')
        
            df[col] = df[col].astype(str)
            df[col] = df[col].astype(str)
        
            le = LabelEncoder()
            le.fit(list(df[col]))
            df[col] = le.transform(df[col])
            df[col] = df[col].astype('category')
            
    return df
        
        
        
        
