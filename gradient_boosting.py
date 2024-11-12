import numpy as np

class GradientBoostingModel:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        residuals = y
        for i in range(self.n_estimators):
            stump = DecisionTreeRegressor(max_depth=self.max_depth)
            stump.fit(X, residuals)
            self.models.append(stump)
            predictions = stump.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        # Start with zeros, as initial predictions
        final_predictions = np.zeros(X.shape[0])
        for model in self.models:
            final_predictions += self.learning_rate * model.predict(X)
        return np.round(final_predictions)  # Assuming binary classification (0 or 1)

class DecisionTreeRegressor:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, residuals):
        # Simple decision stump logic
        self.split_feature = np.argmax(np.var(X, axis=0))  # Split on feature with max variance
        self.threshold = np.median(X[:, self.split_feature])
        left_mask = X[:, self.split_feature] < self.threshold
        right_mask = ~left_mask
        self.left_value = np.mean(residuals[left_mask]) if left_mask.any() else 0
        self.right_value = np.mean(residuals[right_mask]) if right_mask.any() else 0

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        left_mask = X[:, self.split_feature] < self.threshold
        predictions[left_mask] = self.left_value
        predictions[~left_mask] = self.right_value
        return predictions