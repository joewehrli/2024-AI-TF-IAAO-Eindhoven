import numpy as np
from scipy.stats import norm

class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fit(self, X, y):
        if np.isnan(X).any():
            nan_indices = np.where(np.isnan(X))[0]
            nan_rows = X[nan_indices]
            print("Warning arg X has rows with NaN values:\n", nan_rows)
            raise ValueError(f'arg X invalid, has NaN values at: {nan_indices}')
        if np.isnan(y).any():
            nan_indices = np.where(np.isnan(y))[0]
            nan_rows = y[nan_indices]
            print("Warning arg y has rows with NaN values:\n", nan_rows)
            raise ValueError(f'arg y invalid, has NaN values at: {nan_indices}')
        self.X_train = X
        self.y_train = y
        
    def nan_check(np_obj,context):
        if np.isnan(np_obj).any():
            raise ValueError(f'NaN check in np obj {context}')

    def predict(self, X):
        y_pred = []
        index=0
        GRNN.nan_check(self.X_train,'GRNN.self.X_train')
        GRNN.nan_check(X,'X passed to GRNN.predict(...)')
        for x in X:
            try:                
                #calculates the Euclidean distance between x and each row in self.X_train
                diff = self.X_train - x
                GRNN.nan_check(diff,'diff from  self.X_train - x')
                distances = np.linalg.norm(diff, axis=1)
                GRNN.nan_check(distances,'distances from np.linalg.norm(diff, axis=1)')

                #computes the normal distribution's PDF for each distance
                try:
                    weights = norm.pdf(distances / self.sigma)
                    GRNN.nan_check(weights,'weights')
                except ValueError as e:
                    print(f"Error: {e}: weights = norm.pdf(distances / self.sigma), sigma={self.sigma}")    

                #computes the hidden units
                unit_a = np.dot(weights, self.y_train) 
                unit_b = np.sum(weights)
                y_est = -1

                if np.isnan(unit_a):
                    raise ValueError(f'unit_a has NaN')
                if np.isnan(unit_b):
                    raise ValueError(f'unit_b has NaN')
                try:
                    # computes the output unit
                    y_est = unit_a / unit_b
                    y_pred.append(y_est)
                except ValueError as e:
                    print(f"Error: {e}: y_est = unit_a / unit_b, unita={unit_a}, unitb={unit_b}")

                if np.isnan(y_est):
                    raise ValueError(f'y_est has NaN NOTE: y_est = unit_a / unit_b, unita={unit_a}, unitb={unit_b}')
                

            except ValueError as e:
                print(f"Error: {e}")
            index=index+1

        np_y_pred = np.array(y_pred)
        return np_y_pred
