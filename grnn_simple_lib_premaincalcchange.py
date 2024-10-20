import numpy as np
from scipy.stats import norm
from math import isclose
# http://www.statsathome.com/2017/06/26/measure-theory-made-ridiculously-simple/
# https://pieriantraining.com/working-with-scipy-stats-norm-a-guide/

class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def get_params(self, deep=True):
        return {'sigma': self.sigma}
    def set_params(self, **params):
        if 'sigma' in params:
            self.sigma = params['sigma']
        return self
    
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
        
    def adj_dist_calc(x,X_i,sigma):
        diff = X_i - x
        distance_squared = np.dot(diff, diff)    
        adj_dist = distance_squared / sigma
        return adj_dist

    def pi_calc(x,X_i,sigma): #not vectorized, just check one pred
        # Calculate the squared Euclidean distance
        diff = X_i - x
        distance_squared = np.dot(diff, diff)
        # Calculate the Gaussian function
        pi = np.exp(-distance_squared / (2 * sigma ** 2) )
        return pi
    
        
    def predict(self, X):
        y_pred = []
        index=0
        GRNN.nan_check(self.X_train,'GRNN.self.X_train')
        GRNN.nan_check(X,'X passed to GRNN.predict(...)')
        # spot check indices
        check_idx=[0,2]
        for x in X:
            try:                
                #calculates the Euclidean distance between x and each row in self.X_train
                diff = self.X_train - x
                GRNN.nan_check(diff,'diff from  self.X_train - x')
                distances = np.linalg.norm(diff, axis=1)
                GRNN.nan_check(distances,'distances from np.linalg.norm(diff, axis=1)')

                #computes the normal distribution's PDF for each distance
                try:
                    adj_distances = distances / self.sigma
                    weights = norm.pdf(adj_distances)
                    GRNN.nan_check(weights,'weights')
                except ValueError as e:
                    print(f"Error: {e}: weights = norm.pdf(distances / self.sigma), sigma={self.sigma}")    

                if index in check_idx:
                    x_2check=X[index]
                    X_i_2check=self.X_train[0]
                    adjdist_2match=adj_distances[0]
                    adjdist = GRNN.adj_dist_calc(x_2check,X_i_2check,self.sigma)
                    assert np.isclose(adjdist,adjdist_2match), f'adjdist_check failed @ index={index}: check_adjdist: {adjdist} vs algo_adjdist: {adjdist_2match}'

                    pi_2match=weights[0]
                    pi = GRNN.pi_calc(x_2check,X_i_2check,self.sigma)
                    assert np.isclose(pi,pi_2match), f'pi_check failed @ index={index}: check_pi: {pi} vs algo_pi: {pi_2match}'

                #computes the hidden units
                unit_a = np.dot(weights, self.y_train).item()
                unit_b = np.sum(weights).item()

                if np.isnan(unit_a):
                    raise ValueError(f'unit_a has NaN @ index={index}')
                
                if np.isnan(unit_b):
                    raise ValueError(f'unit_b has NaN @ index={index}')
                
                if isclose(unit_a, 0.0) == False and isclose(unit_b, 0.0) == True:
                    raise ValueError(f'unit_b isclose to 0.0 but unit_a is not @ index={index}')
                                
                # computes the output unit
                if isclose(unit_a, 0.0) == True and isclose(unit_b, 0.0) == True:
                    y_est = 0.0 # how to never have this? --> ensure that the weights don't sum to near zero
                else:
                    y_est = unit_a / unit_b
                
                y_pred.append(y_est)   #.item make scalar from np array with one element

                if np.isnan(y_est):
                    raise ValueError(f'y_est has NaN NOTE: y_est = unit_a / unit_b, unita={unit_a}, unitb={unit_b} @ index={index}')

            except ValueError as e:
                print(f"Error: {e}")
            index=index+1

        np_y_pred = np.array(y_pred)
        nz = np.count_nonzero(np_y_pred)
        if nz != len(np_y_pred):
            print(f'Warning all predications are not non-zero: found {len(np_y_pred) - nz} zero values')
        return np_y_pred

class GRNNsearch:
    def __init__(self,):
        __foo=1

    def binary(self, evaluate_fn, low=0.0,high=1.0,tolerance=1e-5):
        """
        Perform a binary search to find the optimal hyperparameter.
        
        Parameters:
        - evaluate_fn: Function that takes a parameter and returns a score.
        - low: The lower bound of the parameter range.
        - high: The upper bound of the parameter range.
        - tolerance: How close we want the search to get before stopping.
        
        Returns:
        - The best parameter value found.
        """
        while (high - low) > tolerance:
            mid1 = low + (high - low) / 3
            mid2 = high - (high - low) / 3
            
            score_mid1 = evaluate_fn(mid1)
            score_mid2 = evaluate_fn(mid2)
            
            if score_mid1 < score_mid2:
                low = mid1  # Narrow search to the upper half
            else:
                high = mid2  # Narrow search to the lower half
            print (f'mid1:{mid1}={score_mid1}')
            print (f'mid2:{mid2}={score_mid2}')

        return (low + high) / 2
    

    