import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from utils import catboost_fillna_pred
from sklearn.metrics import confusion_matrix

model = CatBoostClassifier()
model.load_model('../train_test/catboost_search/db-v3-14_0.02_0.66_31_0.8_2(allin).cbm')
X = joblib.load('../dataset_1st/public_add_features.joblib')
catboost_fillna_pred(X)
y_pred = model.predict_proba(X)[:,1] > 0.46066582604776857

# df31 = pd.read_csv('../submit/31_1124upload.csv')
# df31.set_index('txkey',inplace=True)
# y_pred = df31.loc[X.index,'pred'].to_numpy()

Y = joblib.load('../dataset_2nd/public.joblib')
Y.set_index('txkey',inplace=True)
y_true = Y.loc[X.index,'label'].astype(np.uint8).to_numpy()

C = confusion_matrix(y_true,y_pred)
TN = C[0,0]; FP = C[0,1]
FN = C[1,0]; TP = C[1,1]
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = 2*precision*recall/(precision+recall)

print('F1 score:',f1)