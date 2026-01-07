import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE

# Here I will try to decrease FN in Malicious detections by using: class_weight, AdaBoost, Oversampling.
# Data we use as same for AI_v3

np.set_printoptions(suppress=True)

def print_full_evaluation_report(cm):
    """
    Prints a labeled Confusion Matrix followed by a detailed analysis 
    of the Malicious class performance.
    
    Assumed class order: [0: Non-DoH, 1: Benign, 2: Malicious]
    """
    # Define class labels for the report
    labels = ["Non-DoH", "Benign", "Malicious"]
    
    # 1. PRINT LABELED MATRIX
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    # Print Column Headers (Predictions)
    # The formatting {:>15} aligns text to the right with 15 spaces width
    header = f"{'':>15} | {'Pred: ' + labels[0]:>15} | {'Pred: ' + labels[1]:>15} | {'Pred: ' + labels[2]:>15}"
    print(header)
    print("-" * len(header))
    
    # Print Rows (Actual / True Labels)
    for i, row_label in enumerate(labels):
        row_str = f"{'Act: ' + row_label:>15} | {cm[i, 0]:>15} | {cm[i, 1]:>15} | {cm[i, 2]:>15}"
        print(row_str)
    
    print("-" * len(header))
    
    # 2. PRINT MALICIOUS ANALYSIS (The logic we added earlier)
    # Index 2 represents the Malicious class
    mal_idx = 2
    
    # Retrieve values from the matrix
    tp = cm[mal_idx, 2] # True Positives (Malicious correctly identified)
    fn_nondoh = cm[mal_idx, 0] # Missed: Malicious -> Non-DoH
    fn_benign = cm[mal_idx, 1] # Missed: Malicious -> Benign (Critical)
    total_fn = fn_nondoh + fn_benign
    
    print("\nMALICIOUS TRAFFIC ANALYSIS (Security Focus):")
    print(f"- True Positives: {tp}")
    print(f"- False Negatives (Missed Attacks): {total_fn}")
    print(f"  --> Breakdown: {fn_nondoh} predicted as Non-DoH + {fn_benign} as Benign.")
    print("="*60 + "\n")

def random_forest_model(x_train, x_test, y_train, y_test, feature_names):

    # weights = {0: 1, 1: 1, 2: 8}

    print(f"\n--- Results for: Random Forest (Multiclass) ---")
    
    # 100 trees is a good starting point
    model = RandomForestClassifier(n_estimators=100, random_state=42) 
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Explicitly set labels to ensure [0, 1, 2] order in the matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\n")
    print_full_evaluation_report(cm)
    
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    # Use target_names to make the report readable
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))
    

#AdaBost Model
def adaboost_model(x_train, x_test, y_train, y_test, feature_names):

    print(f"\n--- Results for: Random Forest (Multiclass) ---")
    
    # AdaBoost often uses simple Decision Trees as its base
    base_estimator = DecisionTreeClassifier(max_depth=2) 

    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100, # 100 specialists
        random_state=42
    )


    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Explicitly set labels to ensure [0, 1, 2] order in the matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\n")
    print_full_evaluation_report(cm)
    
    print("\nClassification Report (0=NonDoH, 1=Benign, 2=Malicious):")
    # Use target_names to make the report readable
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['NonDoH (0)', 'Benign (1)', 'Malicious (2)']))
    

#
# --- Main function ---
#
def main():
    
    print("#" * 20)
    print("Multiclass")
    print("#" * 20)

    try:
        df = pd.read_csv('merged_sample.csv')
    except FileNotFoundError:
        print("ERROR: File 'merged_sample.csv' not found!")
        return 

    # Fill any missing values (NaN) with 0
    df.fillna(0, inplace=True)


    x = df.drop(['Label'], axis=1)

    y = df['Label']

    # Print the class distribution
    print(y.value_counts().sort_index())
    print("(0=NonDoH, 1=Benign-DoH, 2=Malicious-DoH)\n")

    # Split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=300)

    
    # print('\nRandom Forest Model Results:')
    # Random_Forest_model(X_train, X_test, y_train, y_test, X.columns)
    # print("-"*100)

    # print('\nRandom Forest Model Results:')
    # AdaBoost_Model(X_train, X_test, y_train, y_test, X.columns)
    # print("-"*100)

    # ---  NEW BLOCK: APPLY SMOTE  ---
    print("\nApplying SMOTE (Oversampling) to the training data...")
    # We only apply this to the TRAINING data, never the TEST data!
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    
    print("Oversampling complete. New training class distribution:")
    print(pd.Series(y_train_resampled).value_counts().sort_index())
    # --- END OF NEW BLOCK ---

    # Now we train our models on the RESAMPLED data
    print('\nRandom Forest Model Results (with SMOTE):')
    random_forest_model(x_train_resampled, x_test, y_train_resampled, y_test, x.columns)
    print("-"*100)
    
    


if __name__ == "__main__":
    main()




# Condition 1:
#
# We are implemented a punishment for wrong choice for model especially for Malicious DoH (class 2)
#
#
# Conclusion 1:
# 
# We see that presents result are worse than previous, and we don`t see any increases`
# Especially here methods of weights does not work
#
#
#Previous Results:
#--- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.12%
# Confusion Matrix (3x3):
# [[ 5837   116    12]
#  [   72 11811    83]
#  [   29   138  5845]]
#
# 5837 - TP for NonDoH
# False Negatives (Missed Attacks): 167 (29 predicted as Non-DoH + 138 as Benign).
#
#Present Results:
# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.10%
# Confusion Matrix (3x3):
# [[ 5840   115    10]
#  [   70 11813    83]
#  [   27   151  5834]]
#
#True Positives: 5834 (Slightly lower than Matrix 1).
#   - False Negatives (Missed Attacks): 178 (27 + 151).
#
# Classification Report (0=NonDoH, 1=Benign, 2=Malicious):
#                precision    recall  f1-score   support
#
#    NonDoH (0)       0.98      0.98      0.98      5965
#    Benign (1)       0.98      0.99      0.98     11966
# Malicious (2)       0.98      0.97      0.98      6012
#
#      accuracy                           0.98     23943
#     macro avg       0.98      0.98      0.98     23943
#  weighted avg       0.98      0.98      0.98     23943
#
#
#
#
# Condition 2:
#
# We use AdaBoost with Random Forest as base estimator
#
#
#
# Conclusion 2: 
#
# The same situation as with weights - the results are worse than previous without AdaBoost
# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 91.99%
# Confusion Matrix (3x3):
# [[ 5574   307    84]
#  [  522 10851   593]
#  [   90   323  5599]]
#   
# True Positives: 5599 (Lowest among all matrices).
#   False Negatives (Missed Attacks): 413 (90 + 323).
#
# Classification Report (0=NonDoH, 1=Benign, 2=Malicious):
#                precision    recall  f1-score   support

#    NonDoH (0)       0.90      0.93      0.92      5965
#    Benign (1)       0.95      0.91      0.93     11966
# Malicious (2)       0.89      0.93      0.91      6012

#      accuracy                           0.92     23943
#     macro avg       0.91      0.92      0.92     23943
#  weighted avg       0.92      0.92      0.92     23943
#
#
#
#
#
#
# Condition 3:
# We use Oversampling (SMOTE) to balance the classes in training data
#
#
# Conclusion 3:
#
# The results are better than previous attempts but accuracy increase a little bit.
#
# Applying SMOTE (Oversampling) to the training data...
# Oversampling complete. New training class distribution:
# Label
# 0    27841
# 1    27841
# 2    27841
# Name: count, dtype: int64

# Random Forest Model Results (with SMOTE):

# --- Results for: Random Forest (Multiclass) ---
# Model Accuracy: 98.19%
# Confusion Matrix (3x3):
# [[ 5863    89    13]
#  [   70 11781   115]
#  [   32   115  5865]]

#- True Positives: 5865 (Highest among all matrices).
#   - False Negatives (Missed Attacks): 147 (32 + 115).

# Classification Report (0=NonDoH, 1=Benign, 2=Malicious):
#                precision    recall  f1-score   support

#    NonDoH (0)       0.98      0.98      0.98      5965
#    Benign (1)       0.98      0.98      0.98     11966
# Malicious (2)       0.98      0.98      0.98      6012

#      accuracy                           0.98     23943
#     macro avg       0.98      0.98      0.98     23943
#  weighted avg       0.98      0.98      0.98     23943
#
#
#
#
# General Conclusion:
# We tested 3 methods to decrease FN for Malicious DoH detection:
# In 2 case (the weights and AdaBoost) the results became worse.
# Only Oversampling (SMOTE) helped to increase the results a bit.
# Generally we need all the calculation because model Random Forest better indicate this parameters