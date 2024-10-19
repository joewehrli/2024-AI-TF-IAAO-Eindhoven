#simple grnn
import pandas as pd
import numpy as np
import grnn_simple_lib as grnn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

combined_grnn_df = pd.read_csv('Output/Eindhoven_Model_Training_Data_Set_GRNN_mlencoded.csv')
Model_Outcome = ['Transaction Price']
xdf = combined_grnn_df.drop(columns=Model_Outcome)
ydf = combined_grnn_df[Model_Outcome].copy()
X_train = xdf.to_numpy()
y_train = ydf.to_numpy()

# Apply StandardScaler to the features
scaler = StandardScaler()
##scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)

##
# GRNN linear form data
ld_grnn_Eindhoven = grnn.GRNN(sigma=0.7507414829659319) #by grid
ld_grnn_Eindhoven.fit(X_train_scaled, y_train)
#ld_grnn_Eindhoven.fit(Eindhoven_data_Train_Feat_linear,
#             Eindhoven_data_Train[Model_Outcome].values.ravel()
#         )
y_pred_train = ld_grnn_Eindhoven.predict(X_train_scaled)
Eindhoven_data_Train = combined_grnn_df
print("mlencoded TRAIN data GRNN R Squared :", r2_score(  ydf[Model_Outcome], y_pred_train ))

##
test_combined_grnn_df = pd.read_csv('Output/Eindhoven_Model_Testing_Data_Set_GRNN_mlencoded.csv')
X_test = test_combined_grnn_df.drop(columns=Model_Outcome).to_numpy()

# Apply the same scaling to the test data
X_test_scaled = scaler.transform(X_test)

y_pred = ld_grnn_Eindhoven.predict(X_test_scaled)

Eindhoven_data_Test = test_combined_grnn_df
print("mlencoded TEST data GRNN R Squared :", r2_score(  Eindhoven_data_Test[Model_Outcome], y_pred ))

"""
#####
# optimize

y_true = Eindhoven_data_Test[Model_Outcome].to_numpy()

def eval_fn(param):
    ld_grnn_Eindhoven = grnn.GRNN(sigma=param)
    ld_grnn_Eindhoven.fit(X_train_scaled, y_train)
    y_pred_eval = ld_grnn_Eindhoven.predict(X_test_scaled)
    res = y_true - y_pred_eval
    mse = np.mean(res**2)
    #rmse = np.sqrt(mse)
    return mse

ld_grnn_search = grnn.GRNNsearch()
search_sigma = ld_grnn_search.binary(eval_fn,low=.001,high=50)
print(search_sigma)

# GRNN linear form data
ld_grnn_Eindhoven = grnn.GRNN(sigma=search_sigma)
ld_grnn_Eindhoven.fit(X_train_scaled, y_train)
#ld_grnn_Eindhoven.fit(Eindhoven_data_Train_Feat_linear,
#             Eindhoven_data_Train[Model_Outcome].values.ravel()
#         )

##
y_pred = ld_grnn_Eindhoven.predict(X_test_scaled)

Eindhoven_data_Test = test_combined_grnn_df
print("mlencoded data GRNN R Squared :", r2_score(  Eindhoven_data_Test[Model_Outcome], y_pred ))
"""


"""
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for sigma
param_grid = {
    'sigma': np.linspace(0.01, 10, 500)  # Testing 500 values of sigma between 0.01 and 5
}

scorer = make_scorer(mean_squared_error, greater_is_better=False)

grid_search = GridSearchCV(grnn.GRNN(), param_grid, scoring=scorer, cv=5)

grid_search.fit(X_train_scaled, y_train)

search_sigma = grid_search.best_params_['sigma']
print(search_sigma)


# GRNN linear form data
ld_grnn_Eindhoven = grnn.GRNN(sigma=search_sigma)
ld_grnn_Eindhoven.fit(X_train_scaled, y_train)
#ld_grnn_Eindhoven.fit(Eindhoven_data_Train_Feat_linear,
#             Eindhoven_data_Train[Model_Outcome].values.ravel()
#         )

##
y_pred = ld_grnn_Eindhoven.predict(X_test_scaled)

Eindhoven_data_Test = test_combined_grnn_df
print("GRID mlencoded TEST data GRNN R Squared :", r2_score(  Eindhoven_data_Test[Model_Outcome], y_pred ))
"""

"""
0.7507414829659319
GRID mlencoded TEST data GRNN R Squared : 0.6315975945497223
"""