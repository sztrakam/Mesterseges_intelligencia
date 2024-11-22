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

missing_columns= [col for col in train.columns if train[col].isnull().sum() > 0]
print(f"Number of missing columns: {missing_columns} ")

print(f"Number of duplicate rows: {train.duplicated().sum()}")

# Removing duplicate rows
train.drop_duplicates(inplace=True)

# Check the shape of the dataset after removing duplicates
print(f"New shape of the dataset: {train.shape}")

print(f"Number of duplicate rows: {test.duplicated().sum()}")

# Removing duplicate rows
test.drop_duplicates(inplace=True)

# Check the shape of the dataset after removing duplicates
print(f"New shape of the dataset: {test.shape}")
sns.countplot(x=train["label"])
df = pd.read_csv(train_url,names=col_names)
numeric_df = test.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='coolwarm')
plt.show()


df = df.copy()
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

if df["src_bytes"].dtype == 'object':
    df["src_bytes"] = pd.to_numeric(df["src_bytes"], errors='coerce')
    df = df.dropna(subset=["src_bytes"])

X = df.drop("src_bytes", axis=1)
y = df["src_bytes"]
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding
X, y = X.sample(frac=0.5, random_state=42), y.sample(frac=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_test = X_test.fillna(0)

model = RandomForestClassifier(random_state=42, n_estimators=10)  ## n_estimators = fák száma, mert alapértelmezetten
##100 fával dolgozik
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model pontossága: {score:.2f}")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=3)  # 5-kép keresztvalidáció
print(f"Cross-validation scores: {scores}")
print(f"Mean score: {scores.mean():.2f}")


##Megnézzük mennyi konstans oszlop található(értékek megegyeznek)
def check_constant_columns(test):
    constant_columns = [col for col in test.columns if test[col].nunique() == 1]
    return constant_columns

# Example usage
constant_cols = check_constant_columns(test)
if constant_cols:
    print(f"Columns with the same value across all rows: {constant_cols}")
else:
    print("No columns have the same value across all rows.")

#Dropping a column because every value is 0
train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

train.describe(include="object")
test.describe(include="object")

from sklearn.preprocessing import LabelEncoder

def LabelEncoding(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

LabelEncoding(train)
LabelEncoding(test)

print(test["protocol_type"].head())

print(len(train)/len(test))

X_train = train.drop(["label"], axis=1)
y_train = train["label"]
X_test = test.drop(["label"], axis=1)
y_test = test["label"]

print(f"{X_train.shape}")
print(f"{X_test.shape}")
print(f"{y_train.shape}")
print(f"{y_test.shape}")


np.random.seed(42)
rfc = RandomForestClassifier()
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)
import itertools
feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]
X_train = X_train[selected_features]
X_test = X_test[selected_features]
np.random.seed(42)
np.array(X_train)
clf = RandomForestClassifier(n_estimators=10, random_state=42)
start_time = time.time()
clf.fit(X_train, y_train.values.ravel())  # Fitting the model
end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")

start_time = time.time()
y_preds = clf.predict(X_test)  # Predicting on the test set
end_time = time.time()
print(f"Testing time: {end_time - start_time:.4f} seconds")


clf = RandomForestClassifier(n_estimators=10)
start_time = time.time()
clf.fit(X_train, y_train.values.ravel())

end_time = time.time()
print("Training time: ", end_time-start_time)

start_time = time.time()
y_preds = clf.predict(X_test)
end_time = time.time()
print("Testing time: ", end_time-start_time)

np.mean(y_preds == y_test)

clf.score(X_train, y_train)

clf.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)

from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_test, y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_preds)


print(f"Accuracy: {accuracy:.2f}")
