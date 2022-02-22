"""Builds a decision tree classifier for iris dataset"""
from __future__ import division
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Node:

    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def gini(y):
    """Calculates gini index, showing how good the split is.

    :param y: List of flower's classes
    :type y: list
    :return: gini index (in range 0, 1)
    :rtype: float
    """
    all_elems = len(y)
    types = [0, 0, 0]
    # print(y)
    for elem in y:
        types[elem] += 1
    gini = 1
    for i in range(len(types)):
        try:
            gini -= (types[i]/all_elems) ** 2
        except ZeroDivisionError:
            continue
    return gini


def get_best_split(X, y):
    """Tests all splits getting the best one on the iteration

    :param X: parameters of flowers
    :type X: list of lists
    :param y: classes of flowers
    :type y: list
    :return: characteristic of the best split (division, feature, left, right)
    :rtype: tuple
    """
    splits = []
    for feature in range(4):
        for i in range(len(X)):
            division = X[i][feature]
            left = []
            right = []
            for j in range(len(y)):
                if X[j][feature] < division:
                    left.append((X[j], y[j]))
                else:
                    right.append((X[j], y[j]))
            left_nums = []
            for elem in left:
                left_nums.append(elem[1])
            right_nums = []
            for elem in right:
                right_nums.append(elem[1])
            ig = gini(y) - gini(left_nums) - gini(right_nums)
            splits.append((ig, division, feature, left, right))
    max_ig = (0, 0)
    for i in range(len(splits)):
        if splits[i][0] > max_ig[0]:
            max_ig = (splits[i][0], i)
    return splits[max_ig[1]][1:]


def one_type_check(y):
    """Checks if elements of the list have the same value"""
    type = y[0]
    for elem in y:
        if elem != type:
            return False
    return True


def get_column(arr, i):
    """Function to convert numpy arr to list"""
    return list(map(lambda x: x[i], arr))


def build_tree(X, y, depth=150) -> Node:
    """Create Nodes recursively, therefore building a decision tree"""
    if depth == 0:
        return y[0]
    if one_type_check(y):
        return y[0]
    threshold, feature, left, right = get_best_split(X, y)
    if not left or not right:
        return y[0]
    return Node(feature, threshold, build_tree(get_column(left, 0),
                get_column(left, 1), depth-1), build_tree(get_column(right, 0),
                get_column(right, 1), depth-1))


def make_predict(root, x_test):
    """Predicts to which class does flower with x_test parameters belond"""
    if isinstance(root, Node):
        if x_test[root.feature] <= root.threshold:
            return make_predict(root.left, x_test)
        else:
            return make_predict(root.right, x_test)
    else:
        return root


def predict(root, x_test_arr):
    """Makes prediction for array of arrays with parameters"""
    return [make_predict(root, x) for x in x_test_arr]


def accuracy(real, expected):
    """Calculates accuracy of prediction"""
    same = 0
    for i in range(len(real)):
        if real[i] == expected[i]:
            same += 1
    return round((same / len(expected)), 3)


if __name__ == '__main__':
    data = load_iris()
    X = data.data
    y = data.target
    X, X_test, y, y_test = train_test_split(X, y, test_size= 0.2)
    node = build_tree(X, y, 4)
    result = predict(node, X_test)
    expected = list(y_test)
    print('Prediction for test:', result)
    print('Real answers:       ', expected)
    print('\nAccuracy:', accuracy(result, expected))
