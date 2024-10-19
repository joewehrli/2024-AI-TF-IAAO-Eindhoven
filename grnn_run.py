#simple grnn
import pandas as pd
import grnn_simple_lib as grnn
from sklearn.metrics import r2_score

combined_grnn_df = pd.read_csv('Output/Eindhoven_Model_Training_Data_Set_GRNN_linear.csv')
Model_Outcome = ['Transaction Price']
xdf = combined_grnn_df.drop(columns=Model_Outcome)
ydf = combined_grnn_df[Model_Outcome].copy()
X_train = xdf.to_numpy()
y_train = ydf.to_numpy()

##
# GRNN linear form data
ld_grnn_Eindhoven = grnn.GRNN(sigma=0.4)
ld_grnn_Eindhoven.fit(X_train, y_train)
#ld_grnn_Eindhoven.fit(Eindhoven_data_Train_Feat_linear,
#             Eindhoven_data_Train[Model_Outcome].values.ravel()
#         )

##

test_combined_grnn_df = pd.read_csv('Output/Eindhoven_Model_Testing_Data_Set_GRNN_linear.csv')
X_test = test_combined_grnn_df.drop(columns=Model_Outcome).to_numpy()
y_pred = ld_grnn_Eindhoven.predict(X_test)

Eindhoven_data_Test = test_combined_grnn_df
print("Linear data GRNN R Squared :", r2_score(  Eindhoven_data_Test[Model_Outcome], y_pred ))