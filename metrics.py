import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

#Generating the confusion matrix!
y_test = pd.read_csv("C:\\MIT\\SEM 5\\SRP\\expected_op.csv")
y_pred = pd.read_csv("C:\\MIT\\SEM 5\\SRP\\predicted_op.csv")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))