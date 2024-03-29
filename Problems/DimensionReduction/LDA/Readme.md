# Дискриминантный анализ

Отличается от PCA тем, что тут будет обучение с учителем.

### Допущения

* Признаки нормально распределены
* Признаки статистически независимы друг от друга
* У классов идентичные матрицы ковариации

### Алгоритм

* Стандартизировать данные
* Для каждого класса вычислить вектор средних.

$$
\vec m_i = \frac{1}{N_i}\sum_{\vec x \in D_i}^{c} \vec x
$$

где $N_i$ - кол. элементов в классе

* Создать матрицы разброса между и внутри классов.   Внутри:

$$
S_W = \sum_{i = 1}^{c} \sum_{\vec x \in D_i}^{c} (\vec x - \vec m_i) * (\vec x - \vec m_i)^T
$$

Между:

$$
S_B = \sum_{i=1}^{c} N_i (\vec m_i - \vec m) * (\vec m_i - \vec m)^T
$$

где $\vec m$ - среднее по всем данным

* Решаем спектральную задачу для $S_W^{-1} S_B$
* Если мы хотим k компонент в итоге, то сортируем собственные пары по собственным числам
и выбираем первые k.
* Теперь из выбранных собственных векторов строим матрицу перехода в новое пространство.
* Делаем переход.

