import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

    # preparing data for stage 1
    y_train_s1 = y_train.copy().replace({2: 1}) # replacing doh classes for one doh class
    # preparing data for stage 2
    mask_s2 = (y_train == 1) | (y_train == 2) # only use training rows where the original label was 1 or 2
    x_train_s2 = x_train[mask_s2]
    y_train_s2 = y_train[mask_s2]

    return [x_train, x_train_s2, x_test, y_train_s1, y_train_s2, y_test, x]

def feature_importance(model, feature_names, stage):
    importances = None

    # for tree-based models (Random Forest, Decision Tree)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'): # for linear models (Logistic Regression, LDA, SGDClassifier)
        importances = np.mean(np.abs(model.coef_), axis=0)

    # if importance scores were found
    if importances is not None:
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})  # create a DataFrame (combine feature names with their importance scores)
        fi_df = fi_df.sort_values(by='Importance', ascending=False) # sort from most important to the least important
        print(f"Top 10 features used by the model for {stage}:"); print(fi_df.head(10)) # print Top 10 most important features
    else:
        print("Feature Importance is not natively available or easily interpretable for this model.")

def roc_curve_plot(model_s1, model_s2, x_test, y_test):
    model_name = model_s1.__class__.__name__

    # check if the model has a decision_function or predict_proba method
    if not (hasattr(model_s1, 'predict_proba')):
        print(f"Skipping ROC plot: Model {model_name} lacks 'predict_proba' or 'decision_function'.")
        return

    # calculate system-wide probabilities ---
    p_s1 = model_s1.predict_proba(x_test) # probabilities from s1 (NonDoH vs DoH)
    prob_nondoh = p_s1[:, 0]       # P(NonDoH)
    prob_doh_branch = p_s1[:, 1]   # P(DoH) -> Pass to Stage 2

    p_s2 = model_s2.predict_proba(x_test) # probabilities from s2 (Benign vs Malicious)
    prob_benign_given_doh = p_s2[:, 0]    # P(Benign | DoH)
    prob_malicious_given_doh = p_s2[:, 1] # P(Malicious | DoH)

    # calculate final scores for each class
    score_0 = prob_nondoh # NonDoH
    score_1 = prob_doh_branch * prob_benign_given_doh # Benign (Must be DoH AND Benign)
    score_2 = prob_doh_branch * prob_malicious_given_doh # Malicious (Must be DoH AND Malicious)
    y_score_system = np.column_stack((score_0, score_1, score_2)) # stacking into a matrix: (n_samples, 3)

    # prepare ground truth (One-vs-Rest)p
    lb = LabelBinarizer()
    lb.fit([0, 1, 2]) # explicitly set classes to ensure [0, 1, 2] order
    y_test_binarized = lb.transform(y_test)
    
    # plotting
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_names = ['NonDoH (0)', 'Benign (1)', 'Malicious (2)']

    plt.figure(figsize=(8, 6)) # single figure

    for i in range(3): # loop through all 3 classes
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_system[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i],
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})',
                 linewidth=2)

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Hierarchical System ROC for {model_name} (OvR)')
    plt.legend(loc="lower right")
    plt.grid(True)

    # save the plot
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    plt.savefig(f"models/hierarchy/roc/{model_name}_{timestamp}.png", dpi=300, bbox_inches='tight')

def save_report(model_s1, model_s2, model_name, training_time, accuracy, cm_df, report, feature_list, importance1, importance2, timestamp):
    # file paths
    base_dir = "models/hierarchy/"
    report_path = f"{base_dir}/reports/{model_name}_h_report_{timestamp}.txt"
    model1_path = f"{base_dir}/{model_name}_s1_h_{timestamp}.joblib"
    model2_path = f"{base_dir}/{model_name}_s2_h_{timestamp}.joblib"
    feats_path = f"{base_dir}/{model_name}_h_features_{timestamp}.joblib"
    header = f"\n--- Results for: {model_name} ---" # header text

    # save the model and feature list
    joblib.dump(model_s1, model1_path)
    joblib.dump(model_s2, model2_path)
    joblib.dump(feature_list, feats_path)

    # open the report file and write/print simultaneously
    with open(report_path, "w") as f:
        
        # helper function to write to both console and file
        def log(text):
            print(text)          # print to console
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
        if (importance1 is not None) and (importance2 is not None):
            log("\nFeature Importance:")
            # check if importance is a dataframe or string before logging
            if hasattr(importance1, 'to_string'): 
                log(importance1.to_string())
                log(importance2.to_string())
            else:
                log(str(importance1))
                log(str(importance2))
        else:
            log("\nSkipping Feature Importance (Not applicable to this model class).")
        
        log(f"Models saved as '{model1_path}', '{model2_path}'")
        log(f"Feature list saved as '{feats_path}'")

    print(f"Report text file saved to: {report_path}")

def model_training(model_s1, model_s2, x_train, x_train_s2, x_test, y_train, y_train_s2, y_test, feature_names):
    # Stage 1: Class 0(Non-DoH) vs {Class 1, 2} (DoH)
    # Stage 2: Class 1 (Benign) vs Class 2 (Malicious)

    start_time = time.time() # starting timer
    model_s1.fit(x_train, y_train) # training the first model (DoH vs Non-DoH)
    model_s2.fit(x_train_s2, y_train_s2) # training the second model (Benign vs Malicious)
    end_time = time.time() # stopping the timer
    model_name = model_s1.__class__.__name__
    feature_list = list(feature_names)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    training_time = end_time - start_time # time spent on training

    pred_s1 = model_s1.predict(x_test) # predict stage 1 (DoH or not?)
    y_pred = np.zeros(len(x_test), dtype=int) # create final prediction array (default to 0)
    
    # for every sample predicted as DoH (1), run stage 2 model
    doh_indices = np.where(pred_s1 == 1)[0]
    if len(doh_indices) > 0:
        x_test_doh = x_test.iloc[doh_indices]
        preds_s2 = model_s2.predict(x_test_doh)
        
        # map stage 2 results back to the final prediction list
        for i, idx in enumerate(doh_indices):
            y_pred[idx] = preds_s2[i]

    # setting matrix and report
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]) # set labels to ensure [0, 1, 2] order in the matrix)
    cm_df = pd.DataFrame(cm, index=['Actual: NonDoH (0)', 'Actual: Benign (1)', 'Actual: Malicious (2)'], # setting up matrix
                             columns=['Pred: NonDoH (0)', 'Pred: Benign (1)', 'Pred: Malicious (2)'])
    report = classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)'])

    # feature importance if supported
    if hasattr(model_s1, 'feature_importances_') or hasattr(model_s1, 'coef_'):
        importance1 = feature_importance(model_s1, feature_names, "Stage 1 (Non-DoH vs DoH)")
        importance2 = feature_importance(model_s2, feature_names, "Stage 2 (Benign vs Malicious)")
    else:
        importance1 = None
        importance2 = None

    save_report(model_s1, model_s2, model_name, training_time, accuracy, cm_df, report, feature_list, importance1, importance2, timestamp)

    roc_curve_plot(model_s1, model_s2, x_test, y_test)
    print("-" * 100)

def main():

    data = prepare_data() # preparing data
    # data has x_train, x_train_s2, x_test, y_train_s1, y_train_s2, y_test, x in given order

    # defining classifiers
    models = [DecisionTreeClassifier(random_state=42),
              RandomForestClassifier(n_estimators=100, random_state=42),
              KNeighborsClassifier(n_neighbors=10)]

    for model in models:
        model_s1 = clone(model)
        model_s2 = clone(model)
        model_training(model_s1, model_s2, data[0], data[1], data[2], data[3], data[4], data[5], data[6].columns) # training and printing results out

if __name__ == "__main__":
    main()
