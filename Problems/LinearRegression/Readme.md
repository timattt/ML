# Регрессионый анализ

Предсказываем сначение непрерывной величины.
Самый простой пример - МНК.

## Seaborn

С помощью библиотеки seaborn можно быстро строить графики зависимости признаков друг от друга.

## Матрицы

### Ковариационная матрица

$$
cov(
\begin{pmatrix}
x_1 \\
x_2
\end{pmatrix},
\begin{pmatrix}
y_1 \\
y_2
\end{pmatrix}
) =
\begin{pmatrix}
\sigma_{x_1 y_1} & \sigma{x_1 y_2} \\
\sigma_{x_2 y_1} & \sigma{x_2 y_2}
\end{pmatrix}
$$

где

$$
\sigma_{x_i y_j} = \overline{(x_i - \overline{x_i})(y_j - \overline{y_j})}
$$

Показывает распределение векторов.

### Корреляционная матрица

Аналогично предыдущей, только компоненты нормированны. И по модулю меньше единицы.
Если коэф. корреляции равен единице, значит есть линейная зависимость.

$$
\sigma_{xy}' = \frac{\sigma_{xy}}{\sigma_{x} \sigma_{y}}
$$