# Bagging

Тут все аналогично мажоритарному голосованию, кроме того, что
тестовые выборки из исходных тренировочных данных выбираются случайно с
возвратом.

![image](https://user-images.githubusercontent.com/25401699/223383637-6750f2c5-b569-479a-be84-0143ef0d0f81.png)

## Тестирование

![image](https://user-images.githubusercontent.com/25401699/223384321-b53a777c-c689-4d6b-8211-3a11e24c1b2a.png)

Точность бэггинга оказалась немного выше.

## Код

```
bag = BaggingClassifier(estimator=tree, n_estimators = 500, max_samples=1.0, max_features=1.0,bootstrap=True, bootstrap_features=False)
```
