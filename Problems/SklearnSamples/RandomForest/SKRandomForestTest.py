from sklearn.ensemble import RandomForestClassifier

def test(X, y):
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=2)
    forest.fit(X, y)
    return forest