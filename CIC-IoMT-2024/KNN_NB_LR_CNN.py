import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, cohen_kappa_score, matthews_corrcoef,  roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle
import glob
import os
from sklearn.utils import shuffle
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

#============================================================
#Import and Pre-processing
#============================================================

# Paths to folder directory with datasets
data_path_test = r"C:\Users\Even\Desktop\Data Analytics and Cyber Intelligence\Machine learning\test"
data_path_train = r"C:\Users\Even\Desktop\Data Analytics and Cyber Intelligence\Machine learning\train"

# Map each filename to one the four categories
label_mapping_train = {
    "ARP_Spoofing_train.pcap.csv": "Spoofing",
    "Benign_train.pcap.csv": "Benign",
    "MQTT-DDoS-Connect_Flood_train.pcap.csv": "MQTT",
    "MQTT-DDoS-Publish_Flood_train.pcap.csv": "MQTT",
    "MQTT-DoS-Connect_Flood_train.pcap.csv": "MQTT",
    "MQTT-DoS-Publish_Flood_train.pcap.csv": "MQTT",
    "MQTT-Malformed_Data_train.pcap.csv": "MQTT",
    "Recon-OS_Scan_train.pcap.csv": "Recon",
    "Recon-Ping_Sweep_train.pcap.csv": "Recon",
    "Recon-Port_Scan_train.pcap.csv": "Recon",
    "Recon-VulScan_train.pcap.csv": "Recon",
}

label_mapping_test = {
    "ARP_Spoofing_test.pcap.csv": "Spoofing",
    "Benign_test.pcap.csv": "Benign",
    "MQTT-DDoS-Connect_Flood_test.pcap.csv": "MQTT",
    "MQTT-DDoS-Publish_Flood_test.pcap.csv": "MQTT",
    "MQTT-DoS-Connect_Flood_test.pcap.csv": "MQTT",
    "MQTT-DoS-Publish_Flood_test.pcap.csv": "MQTT",
    "MQTT-Malformed_Data_test.pcap.csv": "MQTT",
    "Recon-OS_Scan_test.pcap.csv": "Recon",
    "Recon-Ping_Sweep_test.pcap.csv": "Recon",
    "Recon-Port_Scan_test.pcap.csv": "Recon",
    "Recon-VulScan_test.pcap.csv": "Recon",
}

# Read, label, and collect
data_train = []
for filepath in glob.glob(os.path.join(data_path_train, "*.csv")):
    fname = os.path.basename(filepath)
    df = pd.read_csv(filepath, header=0, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop any stray "Unnamed:" columns
    df['Attack_cat'] = label_mapping_train.get(fname, "Unknown")
    data_train.append(df)
# Concatenate all DataFrames
combined = pd.concat(data_train, ignore_index=True)
# Shuffle the combined dataset
data_train = shuffle(combined, random_state=42).reset_index(drop=True)

# Correlation
numeric_cols = data_train.select_dtypes(include=[np.number]).columns.tolist()
corr = data_train[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    #annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.6},
)
plt.title("Feature Correlation Heatmap (Training set)", fontsize=14)
plt.tight_layout()
#plt.show()



# Read, label, and collect
data_test = []
for filepath in glob.glob(os.path.join(data_path_test, "*.csv")):
    fname = os.path.basename(filepath)
    df = pd.read_csv(filepath, header=0, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop any stray "Unnamed:" columns
    df['Attack_cat'] = label_mapping_test.get(fname, "Unknown")
    data_test.append(df)
# Concatenate all DataFrames
combined = pd.concat(data_test, ignore_index=True)
# Shuffle the combined dataset
data_test = shuffle(combined, random_state=42).reset_index(drop=True)

# Check dataset rows
print(f"Train rows : {len(data_train):,}")
print(f"Test  rows : {len(data_test):,}")

print(data_train.head())
print(data_train.info())

print(data_test.head())
print(data_test.info())

# Split ratio
ratio = len(data_test) / (len(data_train) + len(data_test))
print(f"→ explicit split parameter: test ≈ {ratio:.2%}  "
      f"(pre-defined by the dataset provider)")

#print(data_train.head())
#print(data_train.info())


# Encode attack_cat catergories to integral
le = LabelEncoder()
data_train['attack_cat_encoded'] = le.fit_transform(data_train['Attack_cat'])
data_test['attack_cat_encoded'] = le.transform(data_test['Attack_cat'])

# Show the mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label encoding mapping for 'attack_cat':")
for category, code in label_mapping.items():
    print(f"{category}: {code}")

#print(data_train.head())
#print(data_train.info())

# Drop columns
data_train = data_train.drop(['Attack_cat'], axis=1)
data_test = data_test.drop(['Attack_cat'], axis=1)

print(data_train.head())
print(data_train.info())

# checking for missing values
print("Missing values: ", data_train.isna().sum().sum())
print("Missing values: ", data_test.isna().sum().sum())


# Separating target variable and its features
y_train = data_train["attack_cat_encoded"]
X_train = data_train.drop(["attack_cat_encoded"], axis=1)

y_test = data_test["attack_cat_encoded"]
X_test = data_test.drop(["attack_cat_encoded"], axis=1)

# Checks shape
print("x train: ", X_train.shape)
print("x test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)

# For decision tree print
feature_names = X_train.columns.tolist()

# Apply scaling
scaler = StandardScaler()  # or MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#============================================================
#Model training
#============================================================


# Create the classifiers
knn = KNeighborsClassifier()
nb = GaussianNB()
lr = LogisticRegression(max_iter=10000, class_weight='balanced', solver='lbfgs', random_state=42)
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# RNN
n_timesteps = X_train.shape[1]
X_train_rnn = X_train.reshape((X_train.shape[0], n_timesteps, 1))
X_test_rnn  = X_test.reshape((X_test.shape[0],  n_timesteps, 1))
rnn = Sequential([
    SimpleRNN(64, input_shape=(n_timesteps, 1), activation="tanh"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])
rnn.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"])


# Fit the classifiers to the training data
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
lr.fit(X_train, y_train)
rnn.fit(X_train_rnn, y_train_cat, epochs=15, batch_size=256, validation_split=0.1, verbose=2
)

# Predict the labels for the test data
knn_pred = knn.predict(X_test)
nb_pred = nb.predict(X_test)
lr_pred = lr.predict(X_test)
rnn_proba = rnn.predict(X_test_rnn, verbose=0)
rnn_pred  = np.argmax(rnn.predict(X_test_rnn, verbose=0), axis=1)

# Calculate accuracy
knn_acc = accuracy_score(y_test, knn_pred)
nb_acc = accuracy_score(y_test, nb_pred)
lr_acc = accuracy_score(y_test, lr_pred)
rnn_acc = accuracy_score(y_test, rnn_pred)
print(f"KNN Accuracy: {knn_acc:.4f}")
print(f"NB Accuracy: {nb_acc:.4f}")
print(f"LR Accuracy: {lr_acc:.4f}")
print(f"RNN Accuracy: {rnn_acc:.4f}")

print()

# Calculate F1-Score_macro
knn_f1 = f1_score(y_test, knn_pred, average='macro')
nb_f1 = f1_score(y_test, nb_pred, average='macro')
lr_f1 = f1_score(y_test, lr_pred, average='macro')
rnn_f1 = f1_score(y_test, rnn_pred, average='macro')
print(f"KNN F1-Score_macro: {knn_f1:.4f}")
print(f"NB F1-Score_macro: {nb_f1:.4f}")
print(f"LR F1-Score_macro: {lr_f1:.4f}")
print(f"RNN Accuracy: {rnn_f1:.4f}")

print()

# Calculate F1-Score_weighted
knn_f11 = f1_score(y_test, knn_pred, average='weighted')
nb_f11 = f1_score(y_test, nb_pred, average='weighted')
lr_f11 = f1_score(y_test, lr_pred, average='weighted')
rnn_f11 = f1_score(y_test, rnn_pred, average='weighted')
print(f"KNN F1-Score_weighted: {knn_f11:.4f}")
print(f"NB F1-Score_weighted: {nb_f11:.4f}")
print(f"LR F1-Score_weighted: {lr_f11:.4f}")
print(f"RNN F1-Score_weighted: {rnn_f11:.4f}")

print()

# Calculate precision_macro
knn_prec = precision_score(y_test, knn_pred, average='macro')
nb_prec = precision_score(y_test, nb_pred, average='macro')
lr_prec = precision_score(y_test, lr_pred, average='macro')
rnn_prec = precision_score(y_test, rnn_pred, average='macro')
print(f"KNN precision_macro: {knn_prec:.4f}")
print(f"NB precision_macro: {nb_prec:.4f}")
print(f"LR precision_macro: {lr_prec:.4f}")
print(f"RNN precision_macro: {rnn_prec:.4f}")

print()

# Calculate precision_weighted
knn_prec1 = precision_score(y_test, knn_pred, average='weighted')
nb_prec1 = precision_score(y_test, nb_pred, average='weighted')
lr_prec1 = precision_score(y_test, lr_pred, average='weighted')
rnn_prec1 = precision_score(y_test, rnn_pred, average='weighted')
print(f"KNN precision_weighted: {knn_prec1:.4f}")
print(f"NB precision_weighted: {nb_prec1:.4f}")
print(f"LR precision_weighted: {lr_prec1:.4f}")
print(f"RNN precision_weighted: {rnn_prec1:.4f}")

print()

# Calculate recall_macro
knn_rec = recall_score(y_test, knn_pred, average='macro')
nb_rec = recall_score(y_test, nb_pred, average='macro')
lr_rec = recall_score(y_test, lr_pred, average='macro')
rnn_rec = recall_score(y_test, rnn_pred, average='macro')
print(f"KNN recall_macro: {knn_rec:.4f}")
print(f"NB recall_macro: {nb_rec:.4f}")
print(f"LR recall_macro: {lr_rec:.4f}")
print(f"RNN recall_macro: {rnn_rec:.4f}")

print()

# Calculate recall_weighted
knn_rec1 = recall_score(y_test, knn_pred, average='weighted')
nb_rec1 = recall_score(y_test, nb_pred, average='weighted')
lr_rec1 = recall_score(y_test, lr_pred, average='weighted')
rnn_rec1 = recall_score(y_test, rnn_pred, average='weighted')
print(f"KNN recall_weighted: {knn_rec1:.4f}")
print(f"NB recall_weighted: {nb_rec1:.4f}")
print(f"LR recall_weighted: {lr_rec1:.4f}")
print(f"RNN recall_weighted: {rnn_rec1:.4f}")

print()

# Calculate KAPPA
knn_kappa = cohen_kappa_score(y_test, knn_pred)
nb_kappa = cohen_kappa_score(y_test, nb_pred)
lr_kappa = cohen_kappa_score(y_test, lr_pred)
rnn_kappa = cohen_kappa_score(y_test, rnn_pred)

print(f"KNN kappa: {knn_kappa:.4f}")
print(f"NB kappa: {nb_kappa:.4f}")
print(f"LR kappa: {lr_kappa:.4f}")
print(f"RNN kappa: {rnn_kappa:.4f}")

print()

# Calculate MCC
knn_MCC = matthews_corrcoef(y_test, knn_pred)
nb_MCC = matthews_corrcoef(y_test, nb_pred)
lr_MCC = matthews_corrcoef(y_test, lr_pred)
rnn_MCC = matthews_corrcoef(y_test, rnn_pred)
print(f"KNN MCC: {knn_MCC:.4f}")
print(f"NB MCC: {nb_MCC:.4f}")
print(f"LR MCC: {lr_MCC:.4f}")
print(f"RNN MCC: {rnn_MCC:.4f}")

print()

knn_proba  = knn.predict_proba(X_test)
nb_proba  = nb.predict_proba(X_test)
lr_proba  = lr.predict_proba(X_test)

# Calculate AUC_weighted
knn_AUC = roc_auc_score(y_test, knn_proba, average='weighted', multi_class='ovr')
nb_AUC = roc_auc_score(y_test, nb_proba, average='weighted', multi_class='ovr')
lr_AUC = roc_auc_score(y_test, lr_proba, average='weighted', multi_class='ovr')
rnn_AUC = roc_auc_score(y_test, rnn_proba, average='weighted', multi_class='ovr')
print(f"KNN AUC_weighted: {knn_AUC:.4f}")
print(f"NB AUC_weighted: {nb_AUC:.4f}")
print(f"LR AUC_weighted: {lr_AUC:.4f}")
print(f"RNN AUC_weighted: {rnn_AUC:.4f}")

print()

# Calculate AUC_macro
knn_AUC1 = roc_auc_score(y_test, knn_proba, average='macro', multi_class='ovr')
nb_AUC1 = roc_auc_score(y_test, nb_proba, average='macro', multi_class='ovr')
lr_AUC1 = roc_auc_score(y_test, lr_proba, average='macro', multi_class='ovr')
rnn_AUC1 = roc_auc_score(y_test, rnn_proba, average='macro', multi_class='ovr')
print(f"KNN AUC_macro: {knn_AUC1:.4f}")
print(f"NB AUC_macro: {nb_AUC1:.4f}")
print(f"LR AUC_macro: {lr_AUC1:.4f}")
print(f"RNN AUC_macro: {rnn_AUC1:.4f}")

print()


print("=== KNN Classification Report ===")
print(classification_report(y_test, knn_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== NB Classification Report ===")
print(classification_report(y_test, nb_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== LR Classification Report ===")
print(classification_report(y_test, lr_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== RNN Classification Report ===")
print(classification_report(y_test, rnn_pred, digits=4, target_names=le.classes_, zero_division=0))



# Confusion matrix for each model

models = {
    "KNN" : knn_pred,
    "NB" : nb_pred,
    "LR" : lr_pred,
    "RNN" : rnn_pred,
}

n_models     = len(models)
ncols        = 2
nrows        = int(np.ceil(n_models / ncols))

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(10, 4 * nrows),
    dpi=120
)
axes = axes.flatten()   # make it 1-D for easy indexing

# plot each matrix
for ax, (name, y_pred) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,       # single colorbar later
        values_format="d"     # switch to ".1%" for percentage
    )
    ax.set_title(f"{name}", fontsize=12)
    ax.set_xlabel("")        # remove default label (saves space)
    ax.set_ylabel("")        # remove default label
    ax.tick_params(axis="both", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# add ONE shared color-bar
fig.colorbar(
    disp.im_,               # reuse last image object
    ax=axes,
    fraction=0.015,
    pad=0.01
)

# hide any empty subplot if n_models is odd
for j in range(len(models), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Confusion Matrices – Test Set", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()






# Save models to file
models = {
    "knn": knn,
    "nb": nb,
    "lr": lr,
    "rnn": rnn,

}
base_path = "C:\\Users\\Even\\Desktop\\Data Analytics and Cyber Intelligence\\Machine learning\\models_iomt\\"

for name, model in models.items():
    filename = f"{base_path}{name}.sav"
    pickle.dump(model, open(filename, 'wb'))

