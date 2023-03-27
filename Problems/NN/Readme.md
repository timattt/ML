# Многослойная персептронная сеть

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
