import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from eeg_utils import *
# Assuming wnv_get_files() and cobrad_get_files() are defined as provided
# Also assuming weighted_avg function is defined elsewhere in the user's code

def prepare_data(cases_df, controls_df):
    """Prepare data by combining cases and controls with common features."""
    common_cols = list(set(cases_df.columns) & set(controls_df.columns))
    cases = cases_df[common_cols].copy()
    cases['target'] = 1
    controls = controls_df[common_cols].copy()
    controls['target'] = 0
    data = pd.concat([cases, controls], ignore_index=True)
    data = data.dropna()  # Drop rows with missing values
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def train_classical_ml(X_train, y_train, X_test, y_test):
    """Train and evaluate classical ML models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

    return {'Logistic Regression': (acc_lr, auc_lr), 'Random Forest': (acc_rf, auc_rf)}

class Autoencoder(nn.Module):
    """Simple autoencoder for feature extraction."""
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
    """Train and evaluate the NN classifier on encoded features."""
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
        auc = roc_auc_score(y_test_tensor, y_pred)

    return acc, auc

def main():
    """Main function to process datasets and train models."""
    # Process WNV dataset
    df_wnv, _, _, controls_wnv, df_wnv2, _ = wnv_get_files()
    X_wnv, y_wnv = prepare_data(df_wnv2, controls_wnv)
    X_train_wnv, X_test_wnv, y_train_wnv, y_test_wnv = train_test_split(
        X_wnv, y_wnv, test_size=0.2, random_state=42
    )

    # Classical ML for WNV
    results_wnv_classical = train_classical_ml(X_train_wnv, y_train_wnv, X_test_wnv, y_test_wnv)
    print("Classical ML Results for WNV:")
    for model, (acc, auc) in results_wnv_classical.items():
        print(f"{model}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")

    # NN with Autoencoder for WNV
    autoencoder_wnv, scaler_wnv = train_autoencoder(X_train_wnv)
    X_train_wnv_encoded = encode_features(autoencoder_wnv, X_train_wnv, scaler_wnv)
    X_test_wnv_encoded = encode_features(autoencoder_wnv, X_test_wnv, scaler_wnv)
    acc_wnv_nn, auc_wnv_nn = train_nn_classifier(
        X_train_wnv_encoded, y_train_wnv, X_test_wnv_encoded, y_test_wnv
    )
    print(f"NN with Autoencoder for WNV: Accuracy = {acc_wnv_nn:.4f}, AUC = {auc_wnv_nn:.4f}")

    # Process COBRAD dataset
    df_cobrad, _, _, controls_cobrad, df_cobrad2, _ = cobrad_get_files()
    X_cobrad, y_cobrad = prepare_data(df_cobrad2, controls_cobrad)
    X_train_cobrad, X_test_cobrad, y_train_cobrad, y_test_cobrad = train_test_split(
        X_cobrad, y_cobrad, test_size=0.2, random_state=42
    )

    # Classical ML for COBRAD
    results_cobrad_classical = train_classical_ml(X_train_cobrad, y_train_cobrad, X_test_cobrad, y_test_cobrad)
    print("\nClassical ML Results for COBRAD:")
    for model, (acc, auc) in results_cobrad_classical.items():
        print(f"{model}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")

    # NN with Autoencoder for COBRAD
    autoencoder_cobrad, scaler_cobrad = train_autoencoder(X_train_cobrad)
    X_train_cobrad_encoded = encode_features(autoencoder_cobrad, X_train_cobrad, scaler_cobrad)
    X_test_cobrad_encoded = encode_features(autoencoder_cobrad, X_test_cobrad, scaler_cobrad)
    acc_cobrad_nn, auc_cobrad_nn = train_nn_classifier(
        X_train_cobrad_encoded, y_train_cobrad, X_test_cobrad_encoded, y_test_cobrad
    )
    print(f"NN with Autoencoder for COBRAD: Accuracy = {acc_cobrad_nn:.4f}, AUC = {auc_cobrad_nn:.4f}")

if __name__ == "__main__":
    main()