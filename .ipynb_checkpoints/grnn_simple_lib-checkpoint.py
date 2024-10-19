import numpy as np
from scipy.stats import norm

class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def nan_check(np_obj,context):
        if np.isnan(np_obj).any():
            raise ValueError(f'NaN check in np obj {context}')
        
    def predict(self, X):
        y_pred = []
        index=0
        for x in X:
            try:
                #calculates the Euclidean distance between x and each row in self.X_train
                distances = np.linalg.norm(self.X_train - x, axis=1)
                GRNN.nan_check(distances,'distances')

                #computes the normal distribution's PDF for each distance
                try:
                    weights = norm.pdf(distances / self.sigma)
                    GRNN.nan_check(weights,'weights')
                except ValueError as e:
                    print(f"Error: {e}: weights = norm.pdf(distances / self.sigma), sigma={self.sigma}")    

                unit_a = np.dot(weights, self.y_train) 
                unit_b = np.sum(weights)
                try:
                    y_est = unit_a / unit_b
                    y_pred.append(y_est)
                except ValueError as e:
                    print(f"Error: {e}: y_est = unit_a / unit_b, unitb={unit_b}")     

            except ValueError as e:
                print(f"Error: {e}")
            index=index+1

        np_y_pred = np.array(y_pred)
        return np_y_pred
