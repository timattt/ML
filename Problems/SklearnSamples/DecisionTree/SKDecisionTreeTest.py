from sklearn.tree import DecisionTreeClassifier

def test(X, y):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    tree.fit(X, y)
    return tree