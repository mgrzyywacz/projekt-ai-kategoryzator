import numpy as np
from app.decision_tree import DecisionTree

class TestDecisionTree:
    def test_entropy(self):
        tree = DecisionTree()
        assert tree.entropy([1, 1, 1]) == 0
        assert round(tree.entropy([1, 0, 1, 0]), 10) == 1.0

    def test_fit_and_predict(self):
        x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])

        tree = DecisionTree(max_depth=2)
        tree.fit(x, y)
        predictions = tree.predict(x)

        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)

    def test_accuracy(self):
        x = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        y = np.array([0, 0, 1, 1])

        tree = DecisionTree(max_depth=2)
        tree.fit(x, y)
        acc = tree.accuracy(x, y)

        assert 0 <= acc <= 1
