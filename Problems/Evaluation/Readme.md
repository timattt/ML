# Методы оценки моделей

## Конвеер

Можно объединить несколько преобразователей в конвеер при помощи класс Pipeline.

```
Pipeline([ (NAME, TRANSFORMER) ]).fit(X_train, y_train).predict(X_test)
```

Схематично это выгдядит так:

![image](https://user-images.githubusercontent.com/25401699/222958588-ad12be8e-aa58-462d-a397-a09dbc3ca4b2.png)
