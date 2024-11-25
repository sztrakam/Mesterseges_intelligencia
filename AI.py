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
import streamlit as st
train_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'
test_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'

st.set_page_config(page_title="NSL-KDD Intrusion Detection", layout="wide")

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

st.write('Az edzéskészlet méretei:',train.shape)
st.write('A tesztkészlet méretei:',test.shape)

st.write(f'A train adathalmaz fejlécei: {train.info()}')

st.write('Címkék eloszlása az edzéskészletben:')
st.write(train['label'].value_counts())
st.write()
st.write('Label distribution Test set:')
st.write(test['label'].value_counts())

#Rewriting
train["label"] = train["label"].apply(lambda x: "normal" if x == "normal" else "anomalous")
test["label"] = test["label"].apply(lambda x: "normal" if x == "normal" else "anomalous")

st.write(train["label"].value_counts())
st.write()
st.write(test["label"].value_counts())

missing_columns= [col for col in train.columns if train[col].isnull().sum() > 0]
st.write(f"Hiányzó oszlopok száma: {missing_columns} ")

st.write(f"Duplikált sorok számai:  {train.duplicated().sum()}")

train.drop_duplicates(inplace=True)

st.write(f"(train)Az adathalmaz új alakja: {train.shape}")

st.write(f"Duplikált sorok száma: {test.duplicated().sum()}")

test.drop_duplicates(inplace=True)

st.write(f"(test)Az adathalmaz új alakja: {test.shape}")
sns.countplot(x=train["label"])

df = pd.read_csv(train_url,names=col_names)
numeric_df = test.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='coolwarm')
st.pyplot(fig)

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
st.write(f"Model pontossága: {score:.2f}")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=3)  # 5-kép keresztvalidáció
st.write(f"Keresztvalidációs pontszámok: {scores}")
st.write(f"Átlagos  pontszám: {scores.mean():.2f}")


##Megnézzük mennyi konstans oszlop található(értékek megegyeznek)
def check_constant_columns(test):
    constant_columns = [col for col in test.columns if test[col].nunique() == 1]
    return constant_columns

# Example usage
constant_cols = check_constant_columns(test)
if constant_cols:
    st.write(f"Oszlopok azonos értékekkel az összes sorban: {constant_cols}")
else:
    st.write("Nincs oszlop azonos értékkel a sorok között.")

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

st.write(test["protocol_type"].head())

st.write(f'Az eltérés a train és test adathalmaz között: {len(train)/len(test)}')

X_train = train.drop(["label"], axis=1)
y_train = train["label"]
X_test = test.drop(["label"], axis=1)
y_test = test["label"]

##pl 100,10 -> 100 adatpont és 10 jellemző
st.write("Tanuló adathalmaz (X_train) alakja:", X_train.shape)
st.write("Teszt adathalmaz (X_test) alakja:", X_test.shape)
st.write("Tanuló címkék (y_train) alakja:", y_train.shape)
st.write("Teszt címkék (y_test) alakja:", y_test.shape)



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
clf.fit(X_train, y_train.values.ravel())
end_time = time.time()
st.write(f"Training idő: {end_time - start_time:.4f} mp")

start_time = time.time()
y_preds = clf.predict(X_test)
end_time = time.time()
st.write(f"Testing idő: {end_time - start_time:.4f} mp")


clf = RandomForestClassifier(n_estimators=10)
start_time = time.time()
clf.fit(X_train, y_train.values.ravel())

end_time = time.time()
st.write("Training idő: ", end_time-start_time)

start_time = time.time()
y_preds = clf.predict(X_test)
end_time = time.time()
st.write("Testing idő: ", end_time-start_time)

np.mean(y_preds == y_test)

clf.score(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_preds)

from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_test, y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_preds)


st.write(f"Pontosság: {accuracy:.2f}")

scores = cross_val_score(clf, X_test, y_test)
st.write(f"Átlagos pontszám: {np.mean(scores):.2f}")
st.write(f"Szórás: {np.std(scores):.2f}")

#Make some prediction
y_preds= clf.predict(X_test)

from sklearn.metrics import classification_report

st.write("Osztályozási jelentés: ")
st.write(classification_report(y_test, y_preds))

np.array(y_test)
np.array(y_preds)
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
plt.title('Confusion Matrix')
plt.xlabel('Előrejelzett')
plt.ylabel('Tényleges')
st.pyplot(fig)

# Classification Report
st.write(classification_report(y_test, y_preds))

# Kategóriális oszlopok átalakítása numerikussá
label_encoder = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df[col] = label_encoder.fit_transform(df[col])

# Numerikus adat normalizálása
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Elbow módszer az optimális klaszterszám kiválasztásához
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[numerical_features])
    inertia.append(kmeans.inertia_)

# Elbow-görbe megjelenítése
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Klaszterek száma')
##Az "inertia" azt méri, hogy az egyes adatpontok mennyire távol helyezkednek el a hozzájuk rendelt klaszter középpontjától (centroidjától).
plt.ylabel('Inertia')
plt.show()

# Optimális klaszterszám beállítása az elbow-görbe alapján
optimal_k = 3  # Állítsd be az elbow módszer alapján

# KMeans klaszterezés az optimális klaszterszámmal
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df['cluster'] = kmeans.fit_predict(df[numerical_features])

# Távolságok kiszámítása a legközelebbi centroidtól minden adatpontra (csak numerikus oszlopokkal)
df['distance_to_centroid'] = np.min(kmeans.transform(df[numerical_features]), axis=1)

# Azokat tekintjük anomáliának, amelyek távolsága az átlagnál szignifikánsan nagyobb
distance_threshold = df['distance_to_centroid'].mean() + 2 * df['distance_to_centroid'].std()
anomalies = df[df['distance_to_centroid'] > distance_threshold]

st.write("Összes anomália száma:", anomalies.shape[0])
st.write("Anomáliák: ")
st.write(anomalies)

import pandas as pd

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

st.write("Numerikus oszlopok statisztikai mutatói:")
st.write(df[numerical_features].describe())

# Kategóriás oszlopok gyakoriságai
st.write("\nKategóriás oszlopok gyakorisági eloszlása:")
for col in categorical_features:
    st.write(f"\n{col} oszlop:")
    st.write(df[col].value_counts())

# Eltérések és kilengő adatok a numerikus oszlopok között
st.write("\nEltérések és kilengő adatok a numerikus oszlopok között:")
for col in numerical_features:
    # Szórás és maximum-minimum eltérés
    st.write(f"\n{col} oszlop eltérései:")
    st.write(f"  Szórás: {df[col].std():.2f}")
    st.write(f"  Maximum - Minimum eltérés: {df[col].max() - df[col].min():.2f}")
    st.write(f"  25%-os kvartilis: {df[col].quantile(0.25):.2f}")
    st.write(f"  50%-os kvartilis (median): {df[col].quantile(0.5):.2f}")
    st.write(f"  75%-os kvartilis: {df[col].quantile(0.75):.2f}")
    st.write(f"  IQR (Interquartile Range): {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
    st.write(f"  Skewness (aszimmetria): {df[col].skew():.2f}")
    st.write(f"  Kurtosis: {df[col].kurt():.2f}")

    # IQR alapú kilengők meghatározása
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    if not outliers.empty:
        st.write(f"Kilengő adatok ({col}):")
        st.write(outliers[[col]].head())
    else:
        st.write(f"Nincs kilengő adat a {col} oszlopban.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

outlier_flags = pd.DataFrame(index=df.index)

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

    outlier_flags[col] = outliers.astype(int)  # 1, ha kilengő adat, 0 ha nem

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(outlier_flags, cmap='coolwarm', cbar=False, xticklabels=True, yticklabels=False, annot=False, ax=ax)

ax.set_title("Kilengő adatok hőtérképe", fontsize=16)
ax.set_xlabel("Oszlopok", fontsize=14)
ax.set_ylabel("Adatpontok", fontsize=14)
plt.tight_layout()
st.pyplot(fig)

distance_threshold = df['distance_to_centroid'].mean() + 2 * df['distance_to_centroid'].std()
anomaliak = df[df['distance_to_centroid'] > distance_threshold]

# Az anomáliák száma és a thresholdhoz viszonyított arány
anomalia_count = anomaliak.shape[0]
threshold_ratio = anomalia_count / len(df)  # Az anomáliák aránya a teljes adathalmazhoz képest

anomalia_names = anomaliak.columns

st.write(f"Összes anomália: {anomalia_count}")
st.write(f"Az anomáliák aránya a teljes adathalmazban: {threshold_ratio:.4f}")

anomaliak = anomaliak.drop(columns=['duration'], errors='ignore')

numeric_anomaliak = anomaliak.select_dtypes(include=['number'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(numeric_anomaliak.columns, anomalia_count)
ax.set_xlabel('Anomália oszlopok')
ax.set_ylabel('Darabszám')
ax.set_title('Anomáliák előfordulásának száma az oszlopokban')
ax.set_xticks(range(len(numeric_anomaliak.columns)))
ax.set_xticklabels(numeric_anomaliak.columns, rotation=90)
plt.tight_layout()
st.pyplot(fig)