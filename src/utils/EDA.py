import numpy as np
import matplotlib.pyplot as plt


def plot_dis(col, train_df, test_df, D=None, p=None, bins=100):
    line = col
    if D is not None: line += " D: {}".format(D)
    if p is not None: line += " p: {}".format(p)
    print(line)
    
    plt.figure(figsize=(24, 16))
    plt.subplot(2,2,1)
    test_df[col].hist(bins=bins,density=1)
    train_df[col].hist(bins=bins,density=1)
    plt.title('training data above',fontsize=20)
    plt.subplot(2,2,2)
    train_df[col].hist(bins=bins, density=1)
    test_df[col].hist(bins=bins, density=1)
    plt.title('test data above',fontsize=20)
    plt.subplot(2,2,3)
    cut_point = int(train_df.shape[0]/4)
    train_df.loc[:cut_point,col].hist(bins=bins, density=1)
    train_df.loc[3*cut_point:,col].hist(bins=bins, density=1)
    plt.title('isFraud front vs back',fontsize=20)
    plt.subplot(2,2,4)
    tmp = train_df.loc[:cut_point,[col,'isFraud']].groupby(col)['isFraud'].agg('mean')
    window_tmp = int(np.ceil(tmp.shape[0]/100))
    tmp.rolling(window=window_tmp).mean().fillna(0).plot()
    train_df.loc[cut_point:,[col,'isFraud']].groupby(col)['isFraud'].agg('mean').rolling(
        window=window_tmp).mean().fillna(0).plot()
    plt.title('isFraud front vs back',fontsize=20)
    plt.show()
    
    