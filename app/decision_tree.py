import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def information_gain(self, X_column, y, threshold):
        left_mask = X_column <= threshold
        right_mask = X_column > threshold

        if sum(left_mask) == 0 or sum(right_mask) == 0:
            return 0

        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = sum(left_mask), sum(right_mask)

        child_entropy = (n_left / n) * self.entropy(y[left_mask]) + \
                        (n_right / n) * self.entropy(y[right_mask])

        return parent_entropy - child_entropy

    def best_split(self, X, y):
        best_gain = 0
        best_split = None

        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                gain = self.information_gain(feature_values, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain
                    }
        return best_split

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            return {'leaf': True, 'class': unique_classes[0]}
        if self.max_depth and depth >= self.max_depth:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        if n_samples < self.min_samples_split:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        split = self.best_split(X, y)
        if split is None or split['gain'] == 0:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}

        feature_idx = split['feature_idx']
        threshold = split['threshold']

        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self.build_tree(X, y)
        return self

    def predict_sample(self, node, x):
        if node['leaf']:
            return node['class']
        if x[node['feature_idx']] <= node['threshold']:
            return self.predict_sample(node['left'], x)
        else:
            return self.predict_sample(node['right'], x)

    def predict(self, X):
        X = np.array(X)
        return [self.predict_sample(self.tree, x) for x in X]

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.array(y_pred) == np.array(y))
