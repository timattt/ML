# Методы оценки моделей

## Конвеер

Можно объединить несколько преобразователей в конвеер при помощи класс Pipeline.

```
Pipeline([ (NAME, TRANSFORMER) ]).fit(X_train, y_train).predict(X_test)
```

Схематично это выгдядит так:

![image](https://user-images.githubusercontent.com/25401699/222958588-ad12be8e-aa58-462d-a397-a09dbc3ca4b2.png)

## Перекрестная проверка

Если мы хотим более правильно оценить точность модели на исходных данных, то
можно разбить данные на k блоков и сделать k тестов, в каждом из которых
один из блоков будет тестовым, а остальные учебными.

![image](https://user-images.githubusercontent.com/25401699/222959194-f1181ad0-a5d3-4568-8cfa-e310bac4800c.png)

Есть функция:

```
cross_val_score(pipeline, X_train, y_train, cv = 10)
```

## Кривая обучения

Мы хотим получить зависимость точности модели от размера выборки.
Оптимум должен выглядеть вот так:

![image](https://user-images.githubusercontent.com/25401699/223056659-ee295e7a-3839-43b6-938f-609780c2fa27.png)

На нашем тестовом примере будет так:

![image](https://user-images.githubusercontent.com/25401699/223056796-9e7dcec9-4136-45d8-b710-c1a18d512ad7.png)

Для построения кривой у нас есть функция:

```
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
```

Тут мы передаем в параметры данные, модель и сетку размеров для тестирования, причем тестирование происходит перекрестной проверкой, за что отвечает параметр cv.
