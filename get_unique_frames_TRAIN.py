import pandas as pd
import joblib
from sklearn import svm

model = svm.SVC()
x = pd.read_csv("C:\\MIT\\SEM 5\\SRP\\train_ip.csv")
y = pd.read_csv("C:\\MIT\\SEM 5\\SRP\\train_op.csv")  # classes having 0 and 1

model.fit(x, y.values.ravel())
joblib.dump(model, 'svm(asl)trial.pkl')
