#simple grnn
import pandas as pd
import numpy as np
import grnn_simple_lib as grnn
from sklearn.metrics import r2_score

combined_grnn_df = pd.read_csv('Output/Eindhoven_Model_Training_Data_Set_GRNN_mlencoded.csv')
Model_Outcome = ['Transaction Price']
xdf = combined_grnn_df.drop(columns=Model_Outcome)
ydf = combined_grnn_df[Model_Outcome].copy()
X_train = xdf.to_numpy()
y_train = ydf.to_numpy()

##
# GRNN linear form data
ld_grnn_Eindhoven = grnn.GRNN(sigma=2.540664) #byhand
ld_grnn_Eindhoven.fit(X_train, y_train)
#ld_grnn_Eindhoven.fit(Eindhoven_data_Train_Feat_linear,
#             Eindhoven_data_Train[Model_Outcome].values.ravel()
#         )

##
test_combined_grnn_df = pd.read_csv('Output/Eindhoven_Model_Testing_Data_Set_GRNN_mlencoded.csv')
X_test = test_combined_grnn_df.drop(columns=Model_Outcome).to_numpy()
y_pred = ld_grnn_Eindhoven.predict(X_test)

Eindhoven_data_Test = test_combined_grnn_df
print("mlencoded data GRNN R Squared :", r2_score(  Eindhoven_data_Test[Model_Outcome], y_pred ))

#####
# optimize

y_true = Eindhoven_data_Test[Model_Outcome].to_numpy()

def eval_fn(param):
    ld_grnn_Eindhoven = grnn.GRNN(sigma=param)
    ld_grnn_Eindhoven.fit(X_train, y_train)
    y_pred_eval = ld_grnn_Eindhoven.predict(X_test)
    res = y_true - y_pred_eval
    mse = np.mean(res**2)
    #rmse = np.sqrt(mse)
    return mse

ld_grnn_search = grnn.GRNNsearch()
search_sigma = ld_grnn_search.binary(eval_fn,low=.001,high=50)
print(search_sigma)

# GRNN linear form data
ld_grnn_Eindhoven = grnn.GRNN(sigma=search_sigma)
ld_grnn_Eindhoven.fit(X_train, y_train)
#ld_grnn_Eindhoven.fit(Eindhoven_data_Train_Feat_linear,
#             Eindhoven_data_Train[Model_Outcome].values.ravel()
#         )

##
y_pred = ld_grnn_Eindhoven.predict(X_test)

Eindhoven_data_Test = test_combined_grnn_df
print("mlencoded data GRNN R Squared :", r2_score(  Eindhoven_data_Test[Model_Outcome], y_pred ))
