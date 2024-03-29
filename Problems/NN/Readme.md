# Многослойная персептронная сеть

## Устройство

По сути своей - это несколько перспептронов, упакованные в слои.

![image](https://user-images.githubusercontent.com/25401699/228016175-69796320-ca63-43db-affb-be4d8d7ad6cd.png)

Здесь:

$$
\vec A_1 = \vec x
$$

$$
\vec z_2 = W_1 \vec A_1 + \vec b_1
$$

$$
\vec A_2 = \sigma(\vec z_2)
$$

$$
\vec z_3 = W_2 \vec A_2 + \vec b_2
$$

$$
\vec A_3 = \sigma(\vec z_3)
$$

Соответственно, $\vec x$ - входные признаки, а $\vec A_3$ - выходные вероятности для классификации.
Для простоты количество персептронов во всех слоях одинаково и равно $n$.

## Функция ошибки

Обобщим ошибку из обычного персептрона. То есть логарифмическое правдоподобие.

$$
I(\vec A_3) = -\sum_{i=1}^{n} [ y^i \ln A_3^i + (1 - y^i) \ln(1 - A_3^i) ] 
$$

Здесь $y^i$ - истинные значения для образца, а $A_3^i$ - предсказнное значение.
Функция явно зависит от $\vec A_3$.

## Решение задачи оптимизации

Мы хотим минимизировать функцию ошибки, но параметры схемы - это $W_1$, $W_2$, $\vec b_1$, $\vec b_2$.
Для использования алгоритма градиентного спуска, потребуются градиенты по всем этим переменным.
Очевидно, что из явного вида функции ошибки сразу следуют выражения для 

$$
\frac{\partial I}{\partial A_3^k} = \frac{A_3^k - y_k}{A_3^k (1 - A_3^k)}
$$

## Алгоритм обратного распространения ошибки

Теперь нужно научится быстро считать нужные частные производные.

![image](https://user-images.githubusercontent.com/25401699/228018540-9659f253-d390-47c4-941c-f122388f2324.png)

Рассмотрим один слой. Пусть известно $\frac{\partial I}{\partial \vec R}$.
Тогда нужно найти: $\frac{\partial I}{\partial W}$, $\frac{\partial I}{\partial \vec b}$, $\frac{\partial I}{\partial \vec P}$.

Имеем:

$$
\vec Q = W \vec P + \vec b
$$

$$
\vec R = \sigma(\vec Q)
$$

В данном случае будем считать все векторы строками, а не столбцами.

1. Используем прямую связь.

$$
\frac{\partial I}{\partial Q_i} = \frac{\partial I}{\partial R_i} \frac{\partial R_i}{\partial Q_i} = \frac{\partial I}{\partial R_i} \sigma'(Q_i)
$$

2. Учитываем перекрестную связь.

$$
\frac{\partial I}{\partial P_i} = \sum_{k=1}^n \frac{\partial I}{\partial Q_k} \frac{\partial Q_k}{\partial P_i}
$$

3. Берем сложную производную.

$$
\frac{\partial I}{\partial W_{ij}} = \sum_{k=1}^n \frac{\partial I}{\partial Q_k} \frac{\partial Q_k}{\partial W_{ij}} =
\frac{\partial I}{\partial Q_i} P_j
$$

Во-втором переходе учли, что 

$$
\frac{\partial Q_k}{\partial W_{ij}} =
\frac{\partial}{\partial W_{ij}} ( W \vec P + \vec b )_k =
\frac{\partial}{\partial W_{ij}} ( \sum_{m = 1}^{n} W_{km} P_m + b_k) = P_j \delta_{ik}
$$

4. Аналогично, но проще:

$$
\frac{\partial I}{\partial b_i} = \sum_{k=1}^n \frac{\partial I}{\partial Q_k} \frac{\partial Q_k}{\partial b_i} =
\sum_{k=1}^n \frac{\partial I}{\partial Q_k} \delta_{ik} = \sum_{k=1}^n \frac{\partial I}{\partial Q_i}
$$

**Получаем 4 фундаментальных соотношения**

$$
\frac{\partial I}{\partial \vec Q} = \frac{\partial I}{\partial \vec R} \odot \sigma'(\vec Q)
$$

$$
\frac{\partial I}{\partial \vec P} = \frac{\partial I}{\partial \vec Q} W^T
$$

$$
\frac{\partial I}{\partial W} = \vec P^T \frac{\partial I}{\partial \vec Q}
$$

$$
\frac{\partial I}{\partial \vec b} = \frac{\partial I}{\partial \vec Q}
$$

## Частный случай

Теперь можно получить градиенты для случая сети с одним скрытым слоем.

Итак, надо найти: $\frac{\partial I}{\partial W_k}$, $\frac{\partial I}{\partial \vec b_k}$, где k = 1, 2.

Пусть

$$
\frac{\partial I}{\partial \vec A_3} = \vec \delta
$$

Используем предыдущие формулы. Тогда имеем для второго слоя:

$$
\frac{\partial I}{\partial z_3^k} = \frac{\partial I}{\partial A_3^k} \sigma'(z_3^k) =
\frac{A_3^k - y_k}{A_3^k (1 - A_3^k)} \sigma(z_3^k) (1 - \sigma(z_3^k)) =
\frac{A_3^k - y_k}{A_3^k (1 - A_3^k)} A_3^k (1 - A_3^k) = A_3^k - y_k = \delta_k
$$

Здесь учли, что $\sigma'(x) = \sigma(x)[1 - \sigma(x)]$ и $\sigma(z_3^k) = A_3^k$

И далее, очевидно:

$$
\frac{\partial I}{\partial W_2} = \vec A_2^T \vec \delta
$$

$$
\frac{\partial I}{\partial \vec b_2} = \vec \delta
$$

$$
\frac{\partial I}{\partial \vec A_2} = \vec \delta * W_2^T
$$

Тогда для первого слоя:

$$
\frac{\partial I}{\partial z_2} = \frac{\partial I}{\partial A_2} \odot \sigma'(z_2) = 
\vec \delta * W_2^T \odot \sigma'(z_2)
$$

$$
\frac{\partial I}{\partial W_1} = \vec A_1^T * \frac{\partial I}{\partial \vec z_2} = \vec A_1^T * \vec \delta * W_2^T \odot \sigma'(z_2)
$$

$$
\frac{\partial I}{\partial b_1} = \frac{\partial I}{\partial \vec z_2} = \vec \delta * W_2^T \odot \sigma'(z_2)
$$

Итого имеем все градиенты.
