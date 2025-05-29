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
from sklearn import svm
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
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

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
svm = SVC(kernel="linear", class_weight="balanced", random_state=42, max_iter=5000)
svm = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
# AdaBoost
ada = AdaBoostClassifier(random_state=42)
# LightGBM
lgbm = LGBMClassifier(class_weight="balanced", objective="multiclass", num_class=len(le.classes_), random_state=42, n_jobs=-1)

# Fit the classifiers to the training data
svm.fit(X_train, y_train)
ada.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# Predict the labels for the test data
svm_pred  = svm.predict(X_test)
ada_pred  = ada.predict(X_test)
lgbm_pred = lgbm.predict(X_test)

# Calculate accuracy
svm_acc  = accuracy_score(y_test, svm_pred)
ada_acc  = accuracy_score(y_test, ada_pred)
lgbm_acc = accuracy_score(y_test, lgbm_pred)
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"ADA Accuracy: {ada_acc:.4f}")
print(f"LGBM Accuracy: {lgbm_acc:.4f}")

print()

# Calculate F1-Score_macro
svm_f1 = f1_score(y_test, svm_pred, average='macro')
ada_f1 = f1_score(y_test, ada_pred, average='macro')
lgbm_f1 = f1_score(y_test, lgbm_pred, average='macro')
print(f"SVM F1-Score_macro: {svm_f1:.4f}")
print(f"ADA F1-Score_macro: {ada_f1:.4f}")
print(f"LGBM F1-Score_macro: {lgbm_f1:.4f}")

print()

# Calculate F1-Score_weighted
# F1 (weighted)
svm_f11  = f1_score(y_test, svm_pred,  average="weighted")
ada_f11  = f1_score(y_test, ada_pred,  average="weighted")
lgbm_f11 = f1_score(y_test, lgbm_pred, average="weighted")
print(f"SVM F1-Score_weighted: {svm_f11:.4f}")
print(f"ADA F1-Score_weighted: {ada_f11:.4f}")
print(f"LGBM F1-Score_weighted: {lgbm_f11:.4f}")

print()

# Calculate precision_macro
svm_prec = precision_score(y_test, svm_pred, average='macro')
ada_prec = precision_score(y_test, ada_pred, average='macro')
lgbm_prec = precision_score(y_test, lgbm_pred, average='macro')
print(f"SVM precision_macro: {svm_prec:.4f}")
print(f"ADA precision_macro: {ada_prec:.4f}")
print(f"LGBM precision_macro: {lgbm_prec:.4f}")

print()

# Calculate precision_weighted
svm_prec1 = precision_score(y_test, svm_pred, average='weighted')
ada_prec1 = precision_score(y_test, ada_pred, average='weighted')
lgbm_prec1 = precision_score(y_test, lgbm_pred, average='weighted')
print(f"SVM precision_weighted: {svm_prec1:.4f}")
print(f"ADA precision_weighted: {ada_prec1:.4f}")
print(f"LGBM precision_weighted: {lgbm_prec1:.4f}")

print()

# Calculate recall_macro
svm_rec = recall_score(y_test, svm_pred, average='macro')
ada_rec = recall_score(y_test, ada_pred, average='macro')
lgbm_rec = recall_score(y_test, lgbm_pred, average='macro')
print(f"SVM recall_macro: {svm_rec:.4f}")
print(f"ADA recall_macro: {ada_rec:.4f}")
print(f"LGBM recall_macro: {lgbm_rec:.4f}")

print()

# Calculate recall_weighted
svm_rec1 = recall_score(y_test, svm_pred, average='weighted')
ada_rec1 = recall_score(y_test, ada_pred, average='weighted')
lgbm_rec1 = recall_score(y_test, lgbm_pred, average='weighted')
print(f"SVM recall_weighted: {svm_rec1:.4f}")
print(f"ADA recall_weighted: {ada_rec1:.4f}")
print(f"LGBM recall_weighted: {lgbm_rec1:.4f}")

print()

# Calculate KAPPA
svm_kappa = cohen_kappa_score(y_test, svm_pred)
ada_kappa = cohen_kappa_score(y_test, ada_pred)
lgbm_kappa = cohen_kappa_score(y_test, lgbm_pred)

print(f"SVM kappa: {svm_kappa:.4f}")
print(f"ADA kappa: {ada_kappa:.4f}")
print(f"LGBM kappa: {lgbm_kappa:.4f}")

print()

# Calculate MCC
svm_MCC = matthews_corrcoef(y_test, svm_pred)
ada_MCC = matthews_corrcoef(y_test, ada_pred)
lgbm_MCC = matthews_corrcoef(y_test, lgbm_pred)
print(f"SVM MCC: {svm_MCC:.4f}")
print(f"ADA MCC: {ada_MCC:.4f}")
print(f"LGBM MCC: {lgbm_MCC:.4f}")

print()

svm_proba  = svm.predict_proba(X_test)
ada_proba  = ada.predict_proba(X_test)
lgbm_proba = lgbm.predict_proba(X_test)

# Calculate AUC_weighted
svm_AUC = roc_auc_score(y_test, svm_proba, average='weighted', multi_class='ovr')
ada_AUC = roc_auc_score(y_test, ada_proba, average='weighted', multi_class='ovr')
lgbm_AUC = roc_auc_score(y_test, lgbm_proba, average='weighted', multi_class='ovr')
print(f"SVM AUC_weighted: {svm_AUC:.4f}")
print(f"ADA AUC_weighted: {ada_AUC:.4f}")
print(f"LGBM AUC_weighted: {lgbm_AUC:.4f}")

print()

# Calculate AUC_macro
svm_AUC1 = roc_auc_score(y_test, svm_proba, average='macro', multi_class='ovr')
ada_AUC1 = roc_auc_score(y_test, ada_proba, average='macro', multi_class='ovr')
lgbm_AUC1 = roc_auc_score(y_test, lgbm_proba, average='macro', multi_class='ovr')
print(f"SVM AUC_macro: {svm_AUC1:.4f}")
print(f"ADA AUC_macro: {ada_AUC1:.4f}")
print(f"LGBM AUC_macro: {lgbm_AUC1:.4f}")

print()


# Confusion matrix for each model

models = {
    "SVM;" : svm_pred,
    "ADA" : ada_pred,
    "LGBM" : lgbm_pred,
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
    "svm" : svm,
    "ada" : ada,
    "lgbm" : lgbm,
}
base_path = "C:\\Users\\Even\\Desktop\\Data Analytics and Cyber Intelligence\\Machine learning\\models_iomt\\"

for name, model in models.items():
    filename = f"{base_path}{name}.sav"
    pickle.dump(model, open(filename, 'wb'))

