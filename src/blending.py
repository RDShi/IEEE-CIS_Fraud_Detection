import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

from scipy.stats import describe
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

%matplotlib inline

sub_xgb0 = pd.read_csv('submission_xgb_0.csv', index_col=0) # - 0.9586
sub_xgb1 = pd.read_csv('submission_xgb.csv', index_col=0) # - 0.9583
sub_lgb0 = pd.read_csv('sub_0.9581', index_col=0) # - 0.9581
sub_lgb1 = pd.read_csv('sub_0.95800.csv', index_col=0) # - 0.9581
sub_lgb2 = pd.read_csv('sub_0.9580.csv', index_col=0) # - 0.9581
sub_lgb3 = pd.read_csv('sub_0.9579', index_col=0) # - 0.9581
sub_lgg4 = pd.read_csv('sub_0.9572', index_col=0) # - 0.9581
sub_lgb5 = pd.read_csv('sub_0.95710.csv', index_col=0) # - 0.9581
sub_lgb6 = pd.read_csv('sub_0.9570', index_col=0) # - 0.9581
sub_1=pd.read_csv('submission_0.9558.csv', index_col=0) # - 0.9558
sub_2=pd.read_csv('../ref/submission_0.9544.csv', index_col=0) # - 0.9544
sub_3=pd.read_csv('../ref/submission-0.9537.csv', index_col=0) # - 0.9537


concat_sub = pd.concat([sub_xgb0, sub_xgb1,
                        sub_lgb0, sub_lgb1, sub_lgb2, sub_lgb3, sub_lgb5, sub_lgb6,
                        sub_1, sub_2, sub_3], axis=1)
cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols

# check correlation
corr = concat_sub[cols].corr()
# corr = concat_sub[cols].corr('spearman')
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plot heatmap of correlation
f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))
_ = sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
                annot=True,fmt='.4f', cbar_kws={"shrink":.2})



# to sub.csv
concat_sub['isFraud']= ((1/10)*concat_sub['m0']
                        +(1/10)*concat_sub['m1']
                        +(1/10)*concat_sub['m2']
                        +(1/10)*concat_sub['m3']
                        +(1/10)*concat_sub['m4']
                        +(1/10)*concat_sub['m5']
                        +(1/10)*concat_sub['m6']
                        +(1/10)*concat_sub['m8']
                        +(1/10)*concat_sub['m9']
                        +(1/10)*concat_sub['m10'])
concat_sub[['isFraud']].to_csv('sub_bless.csv')

