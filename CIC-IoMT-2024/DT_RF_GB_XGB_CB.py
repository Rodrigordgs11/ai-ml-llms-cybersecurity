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
import pickle
import glob
import os
from sklearn.utils import shuffle
from joblib import dump

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
plt.show()



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
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
cb = CatBoostClassifier(verbose=0, random_state=42, loss_function='MultiClass')




# Fit the classifiers to the training data
dt.fit(X_train, y_train)
gb.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
cb.fit(X_train, y_train)


# Save decision tree
tree_text = export_text(dt, feature_names=feature_names)
with open("DT_structure.txt", "w", encoding="utf-8") as f:
    f.write(tree_text)


# Save first random forest tree
# ------------------------------------------------------------------
first_tree = rf.estimators_[0]
rf_tree_text = export_text(first_tree, feature_names=feature_names)
with open("rf_first_tree.txt", "w", encoding="utf-8") as f:
    f.write(rf_tree_text)

# Predict the labels for the test data
dt_pred = dt.predict(X_test)
gb_pred = gb.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
cb_pred = cb.predict(X_test)

# Calculate accuracy
dt_acc = accuracy_score(y_test, dt_pred)
gb_acc = accuracy_score(y_test, gb_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
cb_acc = accuracy_score(y_test, cb_pred)
print(f"DT Accuracy: {dt_acc:.4f}")
print(f"GB Accuracy: {gb_acc:.4f}")
print(f"RF Accuracy: {rf_acc:.4f}")
print(f"XGB Accuracy: {xgb_acc:.4f}")
print(f"CB Accuracy: {cb_acc:.4f}")

print()

# Calculate F1-Score_macro
dt_f1 = f1_score(y_test, dt_pred, average='macro')
gb_f1 = f1_score(y_test, gb_pred, average='macro')
xgb_f1 = f1_score(y_test, xgb_pred, average='macro')
rf_f1 = f1_score(y_test, rf_pred, average='macro')
cb_f1 = f1_score(y_test, cb_pred, average='macro')
print(f"DT F1-Score_macro: {dt_f1:.4f}")
print(f"GB F1-Score_macro: {gb_f1:.4f}")
print(f"XGB F1-Score_macro: {xgb_f1:.4f}")
print(f"RF F1-Score_macro: {rf_f1:.4f}")
print(f"CB F1-Score_macro: {cb_f1:.4f}")

print()

# Calculate F1-Score_weighted
dt_f11 = f1_score(y_test, dt_pred, average='weighted')
gb_f11 = f1_score(y_test, gb_pred, average='weighted')
xgb_f11 = f1_score(y_test, xgb_pred, average='weighted')
rf_f11 = f1_score(y_test, rf_pred, average='weighted')
cb_f11 = f1_score(y_test, cb_pred, average='weighted')
print(f"DT F1-Score_weighted: {dt_f11:.4f}")
print(f"GB F1-Score_weighted: {gb_f11:.4f}")
print(f"XGB F1-Score_weighted: {xgb_f11:.4f}")
print(f"RF F1-Score_weighted: {rf_f11:.4f}")
print(f"CB F1-Score_weighted: {cb_f11:.4f}")

print()

# Calculate precision_macro
dt_prec = precision_score(y_test, dt_pred, average='macro')
gb_prec = precision_score(y_test, gb_pred, average='macro')
rf_prec = precision_score(y_test, rf_pred, average='macro')
xgb_prec = precision_score(y_test, xgb_pred, average='macro')
cb_prec = precision_score(y_test, cb_pred, average='macro')
print(f"DT precision_macro: {dt_prec:.4f}")
print(f"GB precision_macro: {gb_prec:.4f}")
print(f"XGB precision_macro: {xgb_prec:.4f}")
print(f"RF precisione_macro: {rf_prec:.4f}")
print(f"CB precision_macro: {cb_prec:.4f}")

print()

# Calculate precision_weighted
dt_prec1 = precision_score(y_test, dt_pred, average='weighted')
gb_prec1 = precision_score(y_test, gb_pred, average='weighted')
rf_prec1 = precision_score(y_test, rf_pred, average='weighted')
xgb_prec1 = precision_score(y_test, xgb_pred, average='weighted')
cb_prec1 = precision_score(y_test, cb_pred, average='weighted')
print(f"DT precision_weighted: {dt_prec1:.4f}")
print(f"GB precision_weighted: {gb_prec1:.4f}")
print(f"XGB precision_weighted: {xgb_prec1:.4f}")
print(f"RF precision_weighted: {rf_prec1:.4f}")
print(f"CB precision_weighted: {cb_prec1:.4f}")

print()

# Calculate recall_macro
dt_rec = recall_score(y_test, dt_pred, average='macro')
gb_rec = recall_score(y_test, gb_pred, average='macro')
rf_rec = recall_score(y_test, rf_pred, average='macro')
xgb_rec = recall_score(y_test, xgb_pred, average='macro')
cb_rec = recall_score(y_test, cb_pred, average='macro')
print(f"DT recall_macro: {dt_rec:.4f}")
print(f"GB recall_macro: {gb_rec:.4f}")
print(f"XGB recall_macro: {xgb_rec:.4f}")
print(f"RF recall_macro: {rf_rec:.4f}")
print(f"CB recall_macro: {cb_rec:.4f}")

print()

# Calculate recall_weighted
dt_rec1 = recall_score(y_test, dt_pred, average='weighted')
gb_rec1 = recall_score(y_test, gb_pred, average='weighted')
rf_rec1 = recall_score(y_test, rf_pred, average='weighted')
xgb_rec1 = recall_score(y_test, xgb_pred, average='weighted')
cb_rec1 = recall_score(y_test, cb_pred, average='weighted')
print(f"DT recall_weighted: {dt_rec1:.4f}")
print(f"GB recall_weighted: {gb_rec1:.4f}")
print(f"XGB recall_weighted: {xgb_rec1:.4f}")
print(f"RF recall_weighted: {rf_rec1:.4f}")
print(f"CB recall_weighted: {cb_rec1:.4f}")

print()

# Calculate KAPPA
dt_kappa = cohen_kappa_score(y_test, dt_pred)
gb_kappa = cohen_kappa_score(y_test, gb_pred)
rf_kappa = cohen_kappa_score(y_test, rf_pred)
xgb_kappa = cohen_kappa_score(y_test, xgb_pred)
cb_kappa = cohen_kappa_score(y_test, cb_pred)
print(f"DT kappa: {dt_kappa:.4f}")
print(f"GB kappa: {gb_kappa:.4f}")
print(f"XGB kappa: {xgb_kappa:.4f}")
print(f"RF kappa: {rf_kappa:.4f}")
print(f"CB kappa: {cb_kappa:.4f}")

print()

# Calculate MCC
dt_MCC = matthews_corrcoef(y_test, dt_pred)
gb_MCC = matthews_corrcoef(y_test, gb_pred)
rf_MCC = matthews_corrcoef(y_test, rf_pred)
xgb_MCC = matthews_corrcoef(y_test, xgb_pred)
cb_MCC = matthews_corrcoef(y_test, cb_pred)
print(f"DT MCC: {dt_MCC:.4f}")
print(f"GB MCC: {gb_MCC:.4f}")
print(f"XGB MCC: {xgb_MCC:.4f}")
print(f"RF MCC: {rf_MCC:.4f}")
print(f"CB MCC: {cb_MCC:.4f}")

print()

dt_proba  = dt.predict_proba(X_test)
gb_proba  = gb.predict_proba(X_test)
rf_proba  = rf.predict_proba(X_test)
xgb_proba = xgb.predict_proba(X_test)
cb_proba  = cb.predict_proba(X_test)


# Calculate AUC_weighted
dt_AUC = roc_auc_score(y_test, dt_proba, average='weighted', multi_class='ovr')
gb_AUC = roc_auc_score(y_test, gb_proba, average='weighted', multi_class='ovr')
rf_AUC = roc_auc_score(y_test, rf_proba, average='weighted', multi_class='ovr')
xgb_AUC = roc_auc_score(y_test, xgb_proba, average='weighted', multi_class='ovr')
cb_AUC = roc_auc_score(y_test, cb_proba, average='weighted', multi_class='ovr')
print(f"DT AUC_weighted: {dt_AUC:.4f}")
print(f"GB AUC_weighted: {gb_AUC:.4f}")
print(f"XGB AUC_weighted: {xgb_AUC:.4f}")
print(f"RF AUC_weighted: {rf_AUC:.4f}")
print(f"CB AUC_weighted: {cb_AUC:.4f}")

print()

# Calculate AUC_macro
dt_AUC1 = roc_auc_score(y_test, dt_proba, average='macro', multi_class='ovr')
gb_AUC1 = roc_auc_score(y_test, gb_proba, average='macro', multi_class='ovr')
rf_AUC1 = roc_auc_score(y_test, rf_proba, average='macro', multi_class='ovr')
xgb_AUC1 = roc_auc_score(y_test, xgb_proba, average='macro', multi_class='ovr')
cb_AUC1 = roc_auc_score(y_test, cb_proba, average='macro', multi_class='ovr')
print(f"DT AUC_macro: {dt_AUC1:.4f}")
print(f"GB AUC_macro: {gb_AUC1:.4f}")
print(f"XGB AUC_macro: {xgb_AUC1:.4f}")
print(f"RF AUC_macro: {rf_AUC1:.4f}")
print(f"CB AUC_macro: {cb_AUC1:.4f}")

print()


print("=== DT Classification Report ===")
print(classification_report(y_test, dt_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== RF Classification Report ===")
print(classification_report(y_test, rf_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== XGBoost Classification Report ===")
print(classification_report(y_test, xgb_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== GBoost Classification Report ===")
print(classification_report(y_test, gb_pred, digits=4, target_names=le.classes_, zero_division=0))

print("=== CatBoost Classification Report ===")
print(classification_report(y_test, cb_pred, digits=4, target_names=le.classes_, zero_division=0))




# Confusion matrix for each model

models = {
    "DT" : dt_pred,
    "RF" : rf_pred,
    "GB" : gb_pred,
    "XGB": xgb_pred,
    "CB" : cb_pred,
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
    "dt": dt,
    "rf": rf,
    "gb": gb,
    "xgb": xgb,
    "cb": cb
}
base_path = "C:\\Users\\Even\\Desktop\\Data Analytics and Cyber Intelligence\\Machine learning\\models_iomt\\"

for name, model in models.items():
    filename = f"{base_path}{name}.sav"
    pickle.dump(model, open(filename, 'wb'))

