import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file_path):
    creditcard = pd.read_csv(file_path)
    creditcard.drop(['ID'],axis=1,inplace=True)

    cc_default_df = creditcard[creditcard['default.payment.next.month'] == 1]
    cc_nondefault_df = creditcard[creditcard['default.payment.next.month'] == 0]

    correlations = creditcard.corr()
    f , ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlations,annot=True)
    fig = plt.figure(figsize=(20,20))
    sns.countplot(x='AGE',hue='default.payment.next.month',data=creditcard)

    fig = plt.figure(figsize=(20,20))
    plt.subplot(211)
    sns.boxplot(x='SEX',y='LIMIT_BAL',data=creditcard)
    plt.subplot(212)
    sns.boxplot(x='SEX',y='LIMIT_BAL',data=creditcard,showfliers=False)

    return creditcard
