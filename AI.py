from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import time
from sklearn.ensemble import RandomForestClassifier
train_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'
test_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


train = pd.read_csv(train_url,header=None, names = col_names)
test = pd.read_csv(test_url, header=None, names = col_names)

print('Dimensions of the Training set:',train.shape)
print('Dimensions of the Test set:',test.shape)

train.info()

print('Label distribution Training set:')
print(train['label'].value_counts())
print()
print('Label distribution Test set:')
print(test['label'].value_counts())

#Rewriting
train["label"] = train["label"].apply(lambda x: "normal" if x == "normal" else "anomalous")
test["label"] = test["label"].apply(lambda x: "normal" if x == "normal" else "anomalous")

print(train["label"].value_counts())
print()
print(test["label"].value_counts())