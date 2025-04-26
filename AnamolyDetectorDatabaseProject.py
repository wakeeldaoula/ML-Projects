import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from time import sleep # for styling
from colorama import Fore, Back, Style, init # for coloring and styling

init(autoreset=True)  # Automatically reset color after each print

# styling
def print_heading(title):
    print(f"\n{Back.BLUE}{Fore.WHITE}{'='*10} {title} {'='*10}\n")
    sleep(0.8)

def print_subheading(sub):
    print(f"{Fore.CYAN}{Style.BRIGHT}{sub}")
    sleep(0.5)



# Load data function to import CV, TR, and GT files and assign column names

def load_data(uploaded_files):
    data = {}
    for file in uploaded_files:
        if "cv" in file.name.lower():
            data['cv'] = pd.read_csv(file, header=None, names=['cpu_usage', 'memory_usage'])
        elif "tr" in file.name.lower():
            data['tr'] = pd.read_csv(file, header=None, names=['cpu_usage', 'memory_usage'])
        elif "gt" in file.name.lower():
            data['gt'] = pd.read_csv(file, header=None, names=['label'])

    if len(data) != 3:
        raise ValueError("Missing one or more required files: CV, TR, GT")
    return data


# This pipeline is currently unused but defines a standard scaler and a Random Forest Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))  # Classifier will run 100 estimators for better accuracy
])


# Model training function using Isolation Forest

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    contamination = float(y_train.sum()) / len(y_train)
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42,
        verbose=0
    )

    model.fit(X_train)  # Training model using only features (unsupervised)
    return model, X_test, y_test


# Function to visualize anomalies on test data

def plot_anomalies(X, y, model):
    fig, ax = plt.subplots(figsize=(10, 8))
    xx, yy = np.meshgrid(
        np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100),
        np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 100)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap=plt.cm.Blues_r, alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

    normal_points = X[y.values == 0]
    anomalies_points = X[y.values == 1]

    ax.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], c='green', s=20, edgecolor='k', label='Normal')
    ax.scatter(anomalies_points.iloc[:, 0], anomalies_points.iloc[:, 1], c='red', s=30, edgecolor='k', label='Anomaly')

    ax.set_title("Anomaly Detection Results")
    ax.set_xlabel("CPU Usage")
    ax.set_ylabel("Memory Usage")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Function to detect anomalies in unseen input data

def detect_anomalies(new_data, model):
    scores = model.decision_function(new_data)
    preds = model.predict(new_data)
    anomalies = np.where(preds == -1, 1, 0)
    result = new_data.copy()
    result['anomaly'] = anomalies
    result['anomaly_score'] = scores
    return result


# Main logic to load data, train model, evaluate and visualize

if __name__ == "__main__":
    files = [
        open('J:\\4-Fourth Smester\\ADBMS Lab\\project done\\data\\cv_server_data.csv', 'r'),
        open('J:\\4-Fourth Smester\\ADBMS Lab\\project done\\data\\tr_server_data.csv', 'r'),
        open('J:\\4-Fourth Smester\\ADBMS Lab\\project done\\data\\gt_server_data.csv', 'r')
    ]

    data = load_data(files)
    X = pd.concat([data['cv'], data['tr']], axis=0)

    if len(data['gt']) < len(X):
        repeat_times = len(X) // len(data['gt']) + 1
        y = pd.concat([data['gt']] * repeat_times, axis=0).iloc[:len(X)]
    else:
        y = data['gt'].iloc[:len(X)]

    model, X_test, y_test = train_model(X, y)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)

    print_heading("Classification Report")
    print(classification_report(y_test,y_pred))
    sleep(1)

    print_heading("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    sleep(1)


    # === Input Section ===
    print_heading("User Input Section")
    f1 = int(input(f"{Fore.YELLOW}Enter value of CPU Usage: "))
    f2 = int(input(f"{Fore.YELLOW}Enter value of Memory Usage: "))

    X_input = [[f1, f2]]
    prediction = model.predict(X_input)
    prediction_label = 'Anomaly' if prediction[0] == -1 else 'Normal'

    print_subheading(f"\nPrediction for the input is: {Fore.RED + prediction_label if prediction_label == 'Anomaly' else Fore.GREEN + prediction_label}")

    plot_anomalies(X_test, y_test, model)
