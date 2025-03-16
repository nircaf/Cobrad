from eeg_utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer

def plot_model_performance(y_true, y_pred, y_prob, model_name, dataset_name,output_folder):
    """
    Generate and save confusion matrix and ROC curve plots.
    
    Parameters:
    - y_true: True labels (numpy array)
    - y_pred: Predicted labels (numpy array)
    - y_prob: Predicted probabilities (numpy array)
    - model_name: Name of the model (e.g., 'Logistic_Regression')
    - dataset_name: Name of the dataset (e.g., 'WNV')
    """
    # make folder if not exist
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {dataset_name} - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # if cm score > .9
    if accuracy_score(y_true, y_pred) > .9:
        os.makedirs(f'{dataset_name}_figures/ml_plots/{output_folder}', exist_ok=True)
        plt.savefig(f'{dataset_name}_figures/ml_plots/{output_folder}/{dataset_name}_{model_name}_confusion_matrix.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} - {model_name}')
    plt.legend(loc="lower right")
    if roc_auc > .9:
        os.makedirs(f'{dataset_name}_figures/ml_plots/{output_folder}', exist_ok=True)
        plt.savefig(f'{dataset_name}_figures/ml_plots/{output_folder}/{dataset_name}_{model_name}_roc_curve.png')
    plt.close()

def prepare_data_vs_controls(cases_df, controls_df = None, target_col='target'):
    """Prepare data by combining cases and controls with common features."""
    common_cols = list(set(cases_df.columns) & set(controls_df.columns))
    cases = cases_df[common_cols].copy()
    controls = controls_df[common_cols].copy()

    # Filter to include only numeric columns
    numeric_cols = cases.select_dtypes(include=[np.number]).columns
    cases = cases[numeric_cols]
    controls = controls[numeric_cols]

    cases[target_col] = 1
    controls[target_col] = 0
    data = pd.concat([cases, controls], ignore_index=True)
    data = data.dropna()  # Drop rows with missing values
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X, y

def run_data_intra(df_cases,database_name):
    clinical_columns, boxplot_columns = get_clinical_and_boxplot_cols(df_cases)
    # Visualization
    numeric_cols = df_cases.select_dtypes(include=[np.number]).columns
    # Iterate over clinical columns
    for col in clinical_columns:
        df_wnv3 = df_cases[df_cases[col].notna()].copy()
        unique_values = df_wnv3[col].unique()
        print(f'Analyzing {col} with {len(unique_values)} unique values')
        if df_wnv3.shape[0] < 3 or unique_values.shape[0] < 2:
            continue
        if len(unique_values) == 2:  # Check if binary
            # check that there are at least 3 in each group (0,1)
            if len(df_wnv3[df_wnv3[col] == 1]) < 3 or len(df_wnv3[df_wnv3[col] == 0]) < 3:
                continue
            if col == 'sex':
                # if 1 'f' else 'm'
                df_wnv3['Group'] = df_wnv3[col].apply(lambda x: 'f' if x == 1 else 'm')
            elif col == 'sex, 1=male':
                df_wnv3['Group'] = df_wnv3[col].apply(lambda x: 'm' if x == 1 else 'f')
            else:
                # group values based on band if =1, else f'not {band}'
                df_wnv3['Group'] = df_wnv3[col].apply(lambda x: col if x == 1 else f'not {col}')
            y = df_wnv3[col]
            # is cols with EEG in the name of the cols of df_wnv3
            X = df_wnv3[[col for col in df_wnv3.columns if 'EEG' in col]]
            # remove from X cols that are more than have nan values
            X = X.dropna(axis=1,thresh=int(X.shape[0]/2))
            # pad median for the rest
            X = X.fillna(X.median())
            # drop cols all same value
            X = X.loc[:,X.nunique()!=1]
            # Train
            train_w_x_y(X, y, database_name,col)
        # If numeric non-binary
        elif col in numeric_cols:
            pass


def train_classical_ml(X_train, y_train, X_test, y_test, dataset_name,output_folder):
    """Train classical ML models and return metrics and predictions."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    # Plot and save feature importance
    feature_importances = rf.feature_importances_
    features = X_train.columns
    indices = np.argsort(feature_importances)[::-1]

    # Select top N features
    top_n = 10
    top_indices = indices[:top_n]
    top_features = features[top_indices]
    top_importances = feature_importances[top_indices]

    # Get the number of samples in each group
    n_train = len(y_train)
    n_test = len(y_test)
    if acc_rf > .9:
        plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
        plt.title(f"Top 10 Feature Importances (Train N={n_train}, Test N={n_test}, Accuracy={acc_rf:.2f})")
        plt.bar(range(top_n), top_importances, align="center")
        plt.xticks(range(top_n), top_features, rotation=90)
        plt.xlim([-1, top_n])
        plt.tight_layout()  # Adjust layout to ensure everything fits without overlap
        os.makedirs(f'{dataset_name}_figures/ml_plots/{output_folder}', exist_ok=True)
        plt.savefig(f'{dataset_name}_figures/ml_plots/{output_folder}/{dataset_name}_Random_Forest_top_{top_n}_feature_importance.png')
        plt.close()

    return {
        'Logistic_Regression': (acc_lr, auc_lr, y_pred_lr, y_prob_lr),
        'Random_Forest': (acc_rf, auc_rf, y_pred_rf, y_prob_rf)
    }

class Autoencoder(nn.Module):
    """Autoencoder for feature extraction."""
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(X_train, encoding_dim=32, epochs=50, batch_size=32):
    """Train the autoencoder on training data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    autoencoder = Autoencoder(input_dim=X_train.shape[1], encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data in train_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    return autoencoder, scaler

def encode_features(autoencoder, X, scaler):
    """Use the encoder to transform features."""
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        encoded = autoencoder.encoder(X_tensor)
    return encoded.numpy()

class Classifier(nn.Module):
    """Neural network classifier for encoded features."""
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_nn_classifier(X_train_encoded, y_train, X_test_encoded, y_test, epochs=50, batch_size=32):
    """Train NN classifier on encoded features and return metrics and predictions."""
    X_train_tensor = torch.tensor(X_train_encoded, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    classifier = Classifier(input_dim=X_train_encoded.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        y_pred = classifier(X_test_tensor)
        y_pred_class = (y_pred > 0.5).float()
        acc = accuracy_score(y_test_tensor, y_pred_class)
        auc_score = roc_auc_score(y_test_tensor, y_pred)

    return acc, auc_score, y_pred_class.numpy(), y_pred.numpy()

def process_dataset(dataset_name):
    """
    Process the specified dataset ('COBRAD' or 'WNV') including data loading,
    preparation, model training, evaluation, and plotting.
    
    Parameters:
    - dataset_name: String specifying the dataset ('COBRAD' or 'WNV')
    """
    # Data Loading
    if dataset_name == 'WNV':
        df_wnv,patients_folder,control_folder,controls,df_cases,cases_group_name = wnv_get_files()
    elif dataset_name == 'COBRAD':
        df_wnv,patients_folder,control_folder,controls,df_cases,cases_group_name = cobrad_get_files()
    else:
        raise ValueError("Invalid dataset name. Choose 'COBRAD' or 'WNV'.")
    # Data Preparation
    run_data_intra(df_cases,database_name=dataset_name)
    # X, y = prepare_data_vs_controls(df_cases, controls)
    # train_w_x_y(X, y, dataset_name)
    
def train_w_x_y(X, y, dataset_name,output_folder):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Classical ML Models
    results_classical = train_classical_ml(X_train, y_train, X_test, y_test, dataset_name,output_folder)
    for model_name, (acc, auc_score, y_pred, y_prob) in results_classical.items():
        print(f"{model_name} ({dataset_name}): Accuracy = {acc:.4f}, AUC = {auc_score:.4f}")
        plot_model_performance(y_test.values, y_pred, y_prob, model_name, dataset_name,output_folder)

    # Neural Network with Autoencoder
    autoencoder, scaler = train_autoencoder(X_train)
    X_train_encoded = encode_features(autoencoder, X_train, scaler)
    X_test_encoded = encode_features(autoencoder, X_test, scaler)
    acc_nn, auc_nn, y_pred_nn, y_prob_nn = train_nn_classifier(X_train_encoded, y_train, X_test_encoded, y_test)
    print(f"NN_Autoencoder ({dataset_name}): Accuracy = {acc_nn:.4f}, AUC = {auc_nn:.4f}")
    plot_model_performance(y_test.values, y_pred_nn, y_prob_nn, 'NN_Autoencoder', dataset_name,output_folder)

if __name__ == "__main__":
    # Process both datasets
    process_dataset('WNV')
    # process_dataset('COBRAD')