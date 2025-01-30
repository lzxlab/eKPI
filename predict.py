import pandas as pd
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import os
import sys
import numpy as np
from sklearn.utils import shuffle
import joblib
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
from sklearn.model_selection import KFold, cross_val_score,cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import make_scorer
from xgboost import plot_importance
import warnings
from sklearn.metrics import roc_curve, roc_auc_score

# Function to calculate and print model performance scores
def calculate_score(model, Xtest, ytest):
   
    # Predict the values using the trained model
    predicted = model.predict(Xtest)

    # Calculate accuracy score
    acc_score = model.score(Xtest, ytest)

    # Predict probabilities for AUC calculation (only for binary classification)
    y_scores = model.predict_proba(Xtest)[:, 1]

    # Calculate AUC score
    auc_score = roc_auc_score(ytest, y_scores)

    # Calculate F1 score
    f1_score = metrics.f1_score(ytest, predicted)

    # Print classification report and confusion matrix
    print(classification_report(ytest, predicted))
    print(confusion_matrix(ytest, predicted))

    # Print individual scores
    print("Accuracy:", acc_score)
    print("AUC Score:", auc_score)

kinase_name = sys.argv[1]

# Load test data
test_data = pd.read_excel('../Training Data/'+kinase_name+'_test_data_withID.xlsx')
X_test = test_data.drop(columns=['label','KPS'])  # Assuming 'y_test' column is named 'label'
y_test = test_data['label']

# Load the trained model
XGB = joblib.load('../Model/train_model_XGB_'+kinase_name+'.m')

calculate_score(XGB, X_test, y_test)

# Calculate predicted probabilities on the test set
y_scores = XGB.predict_proba(X_test)[:, 1]

# Compute ROC curve parameters (False Positive Rate, True Positive Rate, and thresholds)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (chance line)
plt.xlim([0.0, 1.0])  # Set x-axis limits
plt.ylim([0.0, 1.05])  # Set y-axis limits
plt.xticks(np.arange(0, 1.1, 0.1))  # Custom x-axis tick intervals
plt.yticks(np.arange(0, 1.1, 0.1))  # Custom y-axis tick intervals
plt.xlabel('False Positive Rate')  # Label for x-axis
plt.ylabel('True Positive Rate')  # Label for y-axis
plt.title(kinase_name+' Receiver Operating Characteristic (ROC)')  # Title for the plot
plt.legend(loc='lower right')  # Legend placement

# Save the ROC curve as a PNG image with high resolution
plt.savefig(kinase_name+'_AUC.png', format='png', dpi=1200, bbox_inches='tight')

# Save the ROC curve as a PDF
plt.savefig(kinase_name+'_AUC.pdf', format='pdf', bbox_inches='tight')

# Display the plot
plt.show()


# Adjust the figure size and layout parameters
fig, ax = plt.subplots(figsize=(15, 15))  # Create a subplot with specified size (15x15)
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)  # Adjust the layout margins (left, right, top, bottom)

# Plot feature importance using XGBoost's plot_importance function
plot_importance(XGB, ax=ax, color='lightseagreen')  # Plot feature importance on the specified axis with lightseagreen color
plt.title('Tumor Project Importance in AGC', fontsize=14, fontweight='bold')  # Set the title of the plot with specific font size and weight

# Save the plot as a PNG image with high resolution (1200 dpi)
plt.savefig(kinase_name+'_feature_importance.png', format='png', dpi=1200, bbox_inches='tight')  # Save the figure with tight bounding box

# Save the plot as a PDF
plt.savefig(kinase_name+'_feature_importance.pdf', format='pdf', bbox_inches='tight')  # Save the figure as a PDF with tight bounding box

# Display the plot
plt.show()

# Get feature importance from the trained XGBoost model
feature_importance = XGB.get_booster().get_score(importance_type='weight')  # Retrieve the feature importance scores based on the 'weight' type

# Print the feature importance values
print(feature_importance)
