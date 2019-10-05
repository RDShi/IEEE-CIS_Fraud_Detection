import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import datetime
import gc
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import lightgbm as lgb
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.decomposition import PCA
import tqdm
from scipy.stats import ks_2samp

from utils.EDA import *

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

random_state = 1337

warnings.simplefilter('ignore')
sns.set()

# ## load data
# train_identity = pd.read_csv('../input/train_identity.csv')
# train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

# train_df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
# del train_transaction, train_identity
# gc.collect()

# test_identity = pd.read_csv('../input/test_identity.csv')
# test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

# test_df = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
# del test_transaction, test_identity
# gc.collect()

# df = pd.concat([train_df, test_df])
# del train_df, test_df
# gc.collect()
# df.reset_index(drop=True, inplace=True)
# df.to_feather("../features/restart_data.ftr")

df = pd.read_feather('../features/restart_data.ftr')
print(df.shape)

## basic info
## miss count
df['nulls1'] = df.isna().sum(axis=1)

# fill nan
key_col = 'card1'
reduce_cols = ['addr2', 'card2', 'card3', 'card4', 'card5', 'card6']
df['card6'] = np.where(df['card6']=='debit or credit', np.nan, df['card6'])
df['card6'] = np.where(df['card6']=='charge card', np.nan, df['card6'])
for col in reduce_cols:
    card1_cnt_df = df[[key_col]][~df[col].isna()].groupby(key_col)[key_col].agg(['count']).reset_index()
    print(col)
    temp_df = df.groupby([key_col,col])[col].agg(['count']).reset_index().sort_values(
        by=[key_col,'count']).drop_duplicates(subset=key_col, keep='last').reset_index(drop=True)
    temp_df = temp_df.merge(card1_cnt_df,on=key_col, how='left')
    temp_df['proportions'] = temp_df['count_x']/temp_df['count_y']
    dic_df = temp_df[temp_df['proportions']>0.99]
    dic_df.index = dic_df[key_col].values
    dic_map = dic_df[col].to_dict()
    df[col] = np.where(df[col].isna(), df['card1'].map(dic_map), df[col])

## add timestamp
def add_detail_time(df):
    # create date column
    dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
    startdate = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    df_time = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    df['is_holiday'] = (df_time.dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)    
    df['day_id'] = (df['TransactionDT']/(24*60*60)).astype(int)
    df['month_id'] = ((df_time.dt.year-2017)*12 + df_time.dt.month).astype(int)
    return df

df = add_detail_time(df)
print('df.shape: ',df.shape)

# amt feature
df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
df['TransactionAmt_check'] = (df['TransactionAmt'].astype(int)==df['TransactionAmt']).astype(int)
df['TransactionAmt_log'] = np.log(df['TransactionAmt'])

# hardware feature
def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['browser_type'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe.loc[dataframe['browser_type'].str.contains('Samsung', na=False), 'browser_type'] = 'samsung'
    dataframe.loc[dataframe['browser_type'].str.contains('Generic/Android', na=False), 'browser_type'] = 'android'
    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0].astype('float')
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1].astype('float')
    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1].astype(float)

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
        dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"    
    return dataframe

df = id_split(df)

# freq enconding features
count_encoding_features = ['device_name', 'id_31', 'browser_type', 'id_33', "ProductCD",
                           "card1","card2","card3","card5","addr1","addr2",
                           "P_emaildomain","R_emaildomain",
                           "DeviceType","DeviceInfo"]

def count_encoding(col, df):
    df[col+'_count_full'] = df[col].map(df[col].value_counts(dropna=True))
    return df

for col in count_encoding_features:
    print(col)
    df = count_encoding(col, df)
    
## pca for v
importance_V = ['V258', 'V257', 'V294', 'V317', 'V246', 'V201', 'V70', 'V187', 'V243', 'V308', 'V244', 
                'V283', 'V45', 'V62', 'V91', 'V200', 'V156', 'V313', 'V87', 'V312', 'V69', 'V44', 'V189',
                'V281', 'V30', 'V67', 'V225', 'V323', 'V149', 'V314', 'V54', 'V296', 'V53', 'V324', 'V130',
                'V83', 'V12', 'V143', 'V307', 'V90', 'V266', 'V61', 'V76', 'V280', 'V165', 'V209', 'V310', 
                'V86', 'V285', 'V82']

short_int_V =  ['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V15', 'V16', 
                'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 
                'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V39', 'V40', 'V42', 'V43', 'V46', 'V47', 
                'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V57', 'V58', 'V59', 'V60', 'V61', 
                'V62', 'V63', 'V64', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 
                'V76', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V89', 'V90', 'V91', 'V92', 'V93',
                'V94', 'V98', 'V104', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116',
                'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V141', 'V142', 'V153',
                'V154', 'V169', 'V173', 'V174', 'V175', 'V184', 'V194', 'V195', 'V197', 'V223', 'V240', 'V241',
                'V247', 'V250', 'V251', 'V260', 'V284', 'V286', 'V288', 'V289', 'V297', 'V300', 'V301', 'V302', 
                'V304', 'V325', 'V327', 'V328'] 

long_int_V = ['V37', 'V38', 'V44', 'V45', 'V56', 'V77', 'V78', 'V86', 'V87', 'V95', 'V96', 'V97', 'V99', 'V100',
              'V101', 'V102', 'V103', 'V105', 'V106', 'V138', 'V139', 'V140', 'V143', 'V144', 'V145', 'V146', 
              'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V155', 'V156', 'V157', 'V158', 'V167', 'V168',
              'V170', 'V171', 'V172', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V185',
              'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V196', 'V198', 'V199', 'V200', 
              'V201', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V224', 'V225', 'V226', 'V227', 'V228', 
              'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V242', 
              'V243', 'V244', 'V245', 'V246', 'V248', 'V249', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257',
              'V258', 'V259', 'V261', 'V262', 'V269', 'V279', 'V280', 'V281', 'V282', 'V283', 'V285', 'V287', 
              'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V298', 'V299', 'V303', 'V322', 'V323',
              'V324', 'V326', 'V329', 'V330']

binary_V = ['V1', 'V14', 'V41', 'V65', 'V88', 'V107', 'V305']

numeric_V = ['V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 
             'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V202', 'V203', 'V204', 'V205', 
             'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V263',
             'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
             'V277', 'V278', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315',
             'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 
             'V337', 'V338', 'V339']

group_1_V = ['V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 
             'V297', 'V298', 'V299', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310',
             'V311', 'V312', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']

group_2_V = ['V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107',
             'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 
             'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 
             'V132', 'V133', 'V134', 'V135', 'V136', 'V137']

group_3_V = ['V281', 'V282', 'V283', 'V288', 'V289', 'V296', 'V300', 'V301', 'V313', 'V314', 'V315']

group_4_V = ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
             'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34']

group_5_V = ['V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 
             'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74']

group_6_V = ['V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 
             'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94']

group_7_V = ['V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 
             'V48', 'V49', 'V50', 'V51', 'V52']

group_8_V = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']

group_9_V = ['V220', 'V221', 'V222', 'V227', 'V234', 'V238', 'V239', 'V245', 
             'V250', 'V251', 'V255', 'V256', 'V259', 'V270', 'V271', 'V272']

group_10_V = ['V169', 'V170', 'V171', 'V174', 'V175', 'V180', 'V184', 'V185', 'V188', 'V189', 'V194',
              'V195', 'V197', 'V198', 'V200', 'V201', 'V208', 'V209', 'V210']

group_11_V = ['V167', 'V168', 'V172', 'V173', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 
              'V186', 'V187', 'V190', 'V191', 'V192', 'V193', 'V196', 'V199', 'V202', 'V203', 'V204', 
              'V205', 'V206', 'V207', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216']

group_12_V = ['V217', 'V218', 'V219', 'V223', 'V224', 'V225', 'V226', 'V228', 'V229', 'V230', 'V231', 'V232',
              'V233', 'V235', 'V236', 'V237', 'V240', 'V241', 'V242', 'V243', 'V244', 'V246', 'V247', 'V248', 
              'V249', 'V252', 'V253', 'V254', 'V257', 'V258', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265',
              'V266', 'V267', 'V268', 'V269', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278']

group_13_V = ['V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 
              'V334', 'V335', 'V336', 'V337', 'V338', 'V339']

group_14_V = ['V143', 'V144', 'V145', 'V150', 'V151', 'V152', 'V159', 'V160', 'V164', 'V165', 'V166']

group_15_V = ['V138', 'V139', 'V140', 'V141', 'V142', 'V146', 'V147', 'V148', 'V149', 'V153', 'V154', 
              'V155', 'V156', 'V157', 'V158', 'V161', 'V162', 'V163']

groups_V = [group_1_V, group_2_V, group_3_V, group_4_V, group_5_V, group_6_V, group_7_V, group_8_V, 
             group_9_V, group_10_V, group_11_V, group_12_V, group_13_V, group_14_V, group_15_V]

n_comp = 1
pca = PCA(n_components=n_comp, random_state=random_state)

i = 0
for group_V in groups_V:
    print(i)
    pca_Vx = []
    long_mean_Vx = []
    short_mean_Vx = []
    for Vx in group_V:
        if (Vx in importance_V) or (Vx in binary_V):
            continue
        if Vx in numeric_V:
            pca_Vx.append(Vx)
        if Vx in long_int_V:
            long_mean_Vx.append(Vx)
        if Vx in short_int_V:
            short_mean_Vx.append(Vx)
            
    if len(long_mean_Vx) > 0:
        df['long_V_'+str(i)] = df[long_mean_Vx].mean(axis=1)
    if len(short_mean_Vx) > 0:
        df['short_V_'+str(i)] = df[short_mean_Vx].mean(axis=1)
    if len(pca_Vx) > 0:
        tmp_df = df[pca_Vx]
        tmp_df = np.log(df[pca_Vx])
        tmp_df.replace(-np.inf, np.nan, inplace=True)
        tmp_df = (tmp_df - tmp_df.mean()) / (tmp_df.std())
        tmp_df.replace(np.nan, 0.0, inplace=True)
        df['pca_V_'+str(i)] = pca.fit_transform(tmp_df)
        df['pca_V_'+str(i)] = np.where(df[pca_Vx[0]].isna(), np.nan, df['pca_V_'+str(i)])
        
    i += 1

## save data
about_V = []
for col in df:
    if 'V' in col:
        about_V.append(col)
        
df[about_V].to_feather("../features/df_about_V.ftr")
df.drop(about_V, axis=1).to_feather('../features/basic_info_data.ftr')

