# Sklearn

Полезные методы из библиотеки sklearn

* ``datasets.load_iris()`` - загружает данные о цветах. В объекте есть поля ``data`` и 
``target``
* ``Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size, random_state)`` - рандомно
разбивает данные на тестовые и обучающие
* ``StandardScaler().fit(X).transform(X)`` - возвращает выборку с матож 0 и дисперсией 1.