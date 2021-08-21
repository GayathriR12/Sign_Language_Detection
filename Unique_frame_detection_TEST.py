import joblib
import pandas as pd
import numpy as np

# Loading the model from the pickle file
clf = joblib.load('svm(asl)trial.pkl')
fname = pd.read_csv("C:\\MIT\\SEM 5\\SRP\\test_ip.csv")
#predicting
op = clf.predict(fname)
y_test = pd.read_csv("C:\\MIT\\SEM 5\\SRP\\expected_op.csv")
np.savetxt("C:\\MIT\\SEM 5\\SRP\\predicted_op.csv", op, delimiter=",")
