# data_balancing.py

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def balance_data(X_train, y_train, method='smote'):
    """
    Balances the training data using the specified method.
    """
    if method == 'smote':
        balancer = SMOTE(random_state=42)
    elif method == 'adasyn':
        balancer = ADASYN(random_state=42)
    elif method == 'undersample':
        balancer = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Invalid balancing method. Choose from 'smote', 'adasyn', or 'undersample'.")
    
    X_resampled, y_resampled = balancer.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
