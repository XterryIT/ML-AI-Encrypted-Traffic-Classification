import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Set print options to suppress scientific notation for readability
#np.set_printoptions(suppress=True)

def prepare_data():
    try:
        df = pd.read_csv('data/all_params.csv')
    except FileNotFoundError:
        print("ERROR: File not found!")
        return

    # fill any missing values (NaN) with 0
    df.fillna(0, inplace=True)

    # separating label and statistics
    x = df.drop(['Label'], axis=1)
    y = df['Label']

    # print the class distribution
    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    # split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=300)

    ############ SMOTE ############
    print("\nApplying SMOTE (Oversampling) to the training data...")
    # only apply this to the TRAINING data
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    print("Oversampling complete. New training class distribution:")
    print(pd.Series(y_train_resampled).value_counts().sort_index())

    data = [x_train, x_test, y_train, y_test, x, y]
    smote_data = [x_train_resampled, x_test, y_train_resampled, y_test, x, y]

    return data, smote_data

def feature_importance(model, feature_names):
    importances = None

    importances = None

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.mean(np.abs(model.coef_), axis=0)

    if importances is not None:
        fi_df = pd.DataFrame({'Feature': feature_names,'Importance': importances}).sort_values(by='Importance', ascending=False)
        return fi_df
    
    return None

def roc_curve_plot(model, x_test, y_test, smote):
    model_name = model.__class__.__name__

    # check if the model has a decision_function or predict_proba method
    if not (hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')):
        print(f"Skipping ROC plot: Model {model_name} lacks 'predict_proba' or 'decision_function'.")
        return

    # convert multi-class labels into a binary matrix (OvR)
    # y_test has shape (n_samples,), y_test_binarized will have (n_samples, n_classes)
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    n_classes = y_test_binarized.shape[1]

    # getting the prediction scores (probabilities or decision scores)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(x_test)
    else:  # decision_function for models like SVC
        y_score = model.decision_function(x_test)
        # if decision_function is used, it often needs to be scaled/calibrated
        # for proper probability interpretation, add calibration later

    # calculate ROC and AUC for each class (One-vs-Rest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_names = ['NonDoH (0)', 'Benign (1)', 'Malicious (2)']

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i],
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})',
                 linewidth=2)

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")

    # plot settings
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    if smote:
        plt.title(f'ROC Curve for {model_name} (with SMOTE) (OvR)')
        plt.savefig(f"models/multiclass/roc/{model_name}_smote_{timestamp}.png", dpi=300)
    else:
        plt.title(f'ROC Curve for {model_name} (OvR)')
        plt.savefig(f"models/multiclass/roc/{model_name}_{timestamp}.png", dpi=300)

def save_report(model, model_name, training_time, accuracy, cm_df, report, feature_list, importance, timestamp, smote):
    base_dir = "models/multiclass/"
    if smote:
        # file paths
        report_path = f"{base_dir}/reports/{model_name}_multi_SMOTE_report_{timestamp}.txt"
        model_path = f"{base_dir}/smote/{model_name}_multi_SMOTE_{timestamp}.joblib"
        feats_path = f"{base_dir}/smote/{model_name}_multi_SMOTE_features_{timestamp}.joblib"
        header = f"\n--- Results for: {model_name} (with SMOTE) ---" # header text
    else:
        # file paths
        report_path = f"{base_dir}/reports/{model_name}_multi_report_{timestamp}.txt"
        model_path = f"{base_dir}/{model_name}_multi_{timestamp}.joblib"
        feats_path = f"{base_dir}/{model_name}_multi_features_{timestamp}.joblib"
        header = f"\n--- Results for: {model_name} ---" # header text

    # save the model and feature list
    joblib.dump(model, model_path)
    joblib.dump(feature_list, feats_path)

    # open the report file and write/print simultaneously
    with open(report_path, "w") as f:
        def log(text):
            f.write(text + "\n") # write to file with newline

        # start Logging
        log(header)
        log(f"Time spent on training: {training_time:.2f}s")
        log(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        log("Confusion Matrix: (3x3)")
        log(cm_df.to_string()) 
        log("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
        log(report)

        # feature importance
        if importance is not None:
            log("\nFeature Importance:")
            if hasattr(importance, 'to_string'):
                log(importance.to_string())
            else:
                log(str(importance))
        else:
            log("\nSkipping Feature Importance (Not applicable to this model class).")
        
        log(f"Model saved as '{model_path}'")
        log(f"Feature list saved as '{feats_path}'")

    print(f"Report text file saved to: {report_path}") 

def model_training(model, x_train, x_test, y_train, y_test, feature_names, smote):
    start_time = time.time() # starting timer
    model.fit(x_train, y_train) # training the model
    end_time = time.time() # stopping the timer

    # defining vars for prints and saves
    model_name = model.__class__.__name__
    training_time = end_time - start_time # time spent on training

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]) # set labels to ensure [0, 1, 2] order in the matrix)
    cm_df = pd.DataFrame(cm, index=['Actual: NonDoH (0)', 'Actual: Benign (1)', 'Actual: Malicious (2)'], # setting up matrix
                             columns=['Pred: NonDoH (0)', 'Pred: Benign (1)', 'Pred: Malicious (2)'])
    report = classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']) # setting up report
    importance = feature_importance(model, feature_names)
    feature_list = list(feature_names)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    save_report(model, model_name, training_time, accuracy, cm_df, report, feature_list, importance, timestamp, smote)

    roc_curve_plot(model, x_test, y_test, smote)
    print("-" * 100)

def main():

    data, smote_data = prepare_data() # preparing data
    # data has x_train, x_test, y_train, y_test, x, y and smote_data has x_train_resampled, x_test, y_train_resampled, y_test, x, y in this order
    
    # defining classifiers
    models = [DecisionTreeClassifier(random_state=42),
              RandomForestClassifier(n_estimators=100, random_state=42),
              LogisticRegression(max_iter=15000),
              GaussianNB(),
              SVC(kernel='rbf', random_state=42),
              KNeighborsClassifier(n_neighbors=10),
              LinearDiscriminantAnalysis(),
              QuadraticDiscriminantAnalysis(),
              MLPClassifier()]

    for model in models:
        model_training(model, data[0], data[1], data[2], data[3], data[4].columns, False) # training and printing results out
        model_training(model, smote_data[0], smote_data[1], smote_data[2], smote_data[3], smote_data[4].columns, True) # training and printing results out with SMOTEd data


if __name__ == "__main__":
    main()
