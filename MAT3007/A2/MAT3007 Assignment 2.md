# MAT3007 Assignment 2

[TOC]

## A2.1 

Writing the LP in standard form,
$$
\begin{aligned}
&\min\quad &z\\
&\text{subject to}\quad &-x_{1}-2 x_{2}-3 x_{3}-8 x_{4}=z\\
&& x_{1}-x_{2}+x_{3}+s_1 = 2\\
&& x_{3}-x_{4}+s_2= 1\\
&& 2 x_{2}+3 x_{3}+4 x_{4}+s_3 = 8\\
&& x,s \geq 0.
\end{aligned}
$$

Tableau:
$$
\begin{array}{|c|ccccccc|c|}
\hline 
& x_1 & x_2 & x_3 & x_4 & s_1 & s_2& s_3 & -z_0 \\
\hline

\hline 
-z_0 & -1 & -2 & -3 & -8 & 0 & 0& 0 & 0 \\
\hdashline
s_1 & 1 & -2 & 1 & 0 & 1 & 0 & 0 & 2\\
s_2 & 0 & 0 & 1 & -1 & 0 & 1 & 0 & 1\\
s_3 & 0 & 2 & 3 & \boxed4 & 0 & 0 & 1 & 8\\
\hline
-z_0 & -1 & 2 & 3 & 0 & 0 & 0& 2 & 16\\
\hdashline
s_1 & \boxed1 & -2 & 1 & 0 & 1 & 0 & 0 & 2\\
s_2 & 0 & 1/2 & 7/4 & 0 & 0 & 1 & 1/4 & 3\\
x_4 & 0 & 1/2 & 3/4 & 1 & 0 & 0 & 1/4 & 2\\
\hline
-z_0 & 0 & 0 & 4 & 0 & 1 & 0& 2 & \boxed{18}\\
\hdashline
x_1 & 1 & -2 & 1 & 0 & 1 & 0 & 0 & 2\\
s_2 & 0 & 1/2 & 7/4 & 0 & 0 & 1 & 1/4 & 3\\
x_4 & 0 & 1/2 & 3/4 & 1 & 0 & 0 & 1/4 & 2\\
\hline
\end{array}
$$
Hence the optimal value is $18,$ obtained at
$$
(x_1,x_2,x_3,x_4)=(2,0,0,2).
$$

## A2.2

Rewriting the LP in standard form,
$$
\begin{aligned}
&\min\quad &z\\
&\text{subject to}\quad &x_{1}- x_{2}+x_{3}=z\\
&& -2x_{1}+x_{2}-x_{3}-s_1 = 1\\
&& x_{1}-x_{2}-x_3+s_2= 4\\
&& x_{2}- x_{4}= 0\\
&& x,s \geq 0.
\end{aligned}
$$
The auxiliary problem is then:
$$
\begin{aligned}
&\min\quad &\hat z\\
&\text{subject to}\quad &w_1+w_2+w_3=\hat z\\
&& x_{1}- x_{2}+x_{3}=z\\
&& -2x_{1}+x_{2}-x_{3}-s_1+w_1 = 1\\
&& x_{1}-x_{2}-x_3+s_2+w_2= 4\\
&& x_{2}- x_{4}+w_3= 0\\
&& x,s,w \geq 0.
\end{aligned}
$$
Tableau I:
$$
\begin{array}{|c|ccccccc|c|}
\hline 
& x_1 & x_2 & x_3 & x_4 & s_1 & s_2& w &  \\
\hline

-\hat z_0 & 1 & -1 & 2 & 1 & 1 & -1 & * & -5 \\
-z_0 & 1 & -1 & 1 & 0 & 0 & 0 & * & 0\\
\hdashline
w_1 & -2 & 1 & -1 & 0 & -1 & 0 & * & 1\\
w_2 & 1 & -1 & -1 & 0 & 0 & \boxed1 & * & 4\\
w_3 & 0 & 1 & 0 & -1 & 0 & 0 & * & 0\\
\hline

-\hat z_0 & 2 & -2 & 1 & 1 & 1 & 0 & * & -1 \\
-z_0 & 1 & -1 & 1 & 0 & 0 & 0 & * & 0\\
\hdashline
w_1 & -2 & 1 & -1 & 0 & -1 & 0 & * & 1\\
s_2 & 1 & -1 & -1 & 0 & 0 & 1 & * & 4\\
w_3 & 0 & \boxed1 & 0 & -1 & 0 & 0 & * & 0\\
\hline


-\hat z_0 & 2 & 0 & 1 & -1 & 1 & 0 & * & -1 \\
-z_0 & 1 & 0 & 1 & -1 & 0 & 0 & * & 0\\
\hdashline
w_1 & -2 & 0 & -1 & \boxed1 & -1 & 0 & * & 1\\
s_2 & 1 & 0 & -1 & -1 & 0 & 1 & * & 4\\
x_2 & 0 & 1 & 0 & -1 & 0 & 0 & * & 0\\
\hline

-\hat z_0 & 0 & 0 & 0 & 0 & 0 & 0 & * & 0 \\
-z_0 & -1 & 0 & 0 & 0 & -1 & 0 & * & 1\\
\hdashline
x_4 & \color{red}-2 & 0 & -1 & 1 & -1 & 0 & * & 1\\
s_2 & \color{red}-1 & 0 & -2 & 0 & -1 & 1 & * & 5\\
x_2 & \color{red}-2 & 1 & -1 & 0 & -1 & 0 & * & 1\\
\hline

\end{array}
$$
Already we see by selecting $x_1$ as the entering variable, all the ratios will be non-positive. This means we may increase $x_1$ arbitrarily without ever violating the constraints (simply elevate the existing basic variables accordingly). It follows that $z_\text{min}=-\infty.$ That is, $z$ is unbounded from below.

## A2.3

 Notice that by subtracting the first two constraints, we have
$$
7x_4=0,
$$
which implies $x_4=0.$ And we may eliminate $x_4$ from the LP, as in
$$
\begin{aligned}
&\min\quad &z\\
&\text{subject to}\quad &2x_{1}+3x_{2}-2x_5=z\\
&& x_{1}+3x_{2}+x_5 = 2\\
&& -x_{1}-4x_{2}+3x_3= 1\\
&& x \geq 0.
\end{aligned}
$$
The auxiliary problem:
$$
\begin{aligned}
&\min\quad &\hat z\\
&\text{subject to}\quad &w_1 + w_2 = \hat z \\
&& 2x_{1}+3x_{2}-2x_5=z\\
&& x_{1}+3x_{2}+x_5+w_1 = 2\\
&& -x_{1}-4x_{2}+3x_3+w_2= 1\\
&& x,w \geq 0.
\end{aligned}
$$
Tableau I:
$$
\begin{array}{|c|ccccc|c|}
\hline 
& x_1 & x_2 & x_3 & x_5 & w \\
\hline

-\hat z_0 & 0 & 1 & -3 & -1 & * & -3\\
-z_0 & 2 & 3 & 0 & -2 & * & 0\\
\hdashline
w_1 & 1 & 3 & 0 & 1 & * & 2\\
w_2 & -1 & -4 & \boxed3 & 0 & * & 1\\
\hline

-\hat z_0 & -1 & -3 & 0 & -1 & * & -2\\
-z_0 & 2 & 3 & 0 & -2 & * & 0\\
\hdashline
w_1 & 1 & \boxed3 & 0 & 1 & * & 2\\
x_3 & -1/3 & -4/3 & 1 & 0 & * & 1/3\\
\hline

-\hat z_0 & 0 & 0 & 0 & 0 & * & 0\\
-z_0 & 1 & 0 & 0 & -3 & * & -2\\
\hdashline
x_2 & 1/3 & 1 & 0 & 1/3 & * & 2/3\\
x_3 & 1/9 & 0 & 1 & 4/9 & * & 11/9\\
\hline


\end{array}
$$
Tableau II:
$$
\begin{array}{|c|cccc|c|}
\hline 
& x_1 & x_2 & x_3 & x_5 \\
\hline

-z_0 & 1 & 0 & 0 & -3  & -2\\
\hdashline
x_2 & 1/3 & 1 & 0 & \boxed{1/3}  & 2/3\\
x_3 & 1/9 & 0 & 1 & 4/9  & 11/9\\
\hline

-z_0 & 4 & 9 & 0 & 0  & \boxed4\\
\hdashline
x_5 & 1 & 3 & 0 & 1  & 2\\
x_3 & -1/3 & -4/3 & 1 & 0  & 1/3\\
\hline


\end{array}
$$
Hence the optimal value is $-4,$ obtained at
$$
(x_1,x_2,x_3,x_4,x_5)=(0,0,1/3,0,2).
$$

## A2.4

### (a)

$$
\beta\gt 0.
$$

### (b)

$$
\delta<0;\alpha\le 0;\beta \ge0.
$$
### (c)

$$
\delta \ge0;\beta \ge10.
$$

### (d)

$$
\delta =\eta=\beta=10^{100}.
$$

### (e)

$$
\beta = 0;\eta\ge4;\delta+2\alpha\ge0.
$$

## A2.5

### (a)

We start by writing the objective function as
$$
f(x)=f(x^*)+\bar c_N^\top x_N
$$
with $\bar c_N>0$ and $N$ being the non-basic indices. Let $y$ be any feasible solution other than $x^*$. We claim that $y_N\neq0.$ For, if $y_N =0,$ then since $y$ is feasible,
$$
Ay=A_By_B+A_Ny_N=A_By_B+A_N0=A_By_B=b\implies y_B=A_B^{-1}b=x^*_B.
$$
But then $y_N=x^*_N=0$ forces $y=x^*,$ a contradiction. Therefore $y_N\neq0.$ It follows from the feasibility that $y_n\gt0$ for some non-basic index $i\in N.$ Hence,
$$
f(y)=f(x^*)+\bar c_N^\top y_N\ge f(x^*)+\bar c_iy_i\gt f(x^*),
$$
from which we conclude $x^*$ is the unique optimal solution.

### (b)

Suppose otherwise. Then $\bar c_i\le 0$ for some non-basic index $i.$ Since $x^*$ is nondegenerate, we may incorporate $x_i$ into the basis by taking a sufficiently small step $\theta\gt 0$ in the $i$-th basic direction $d_i$, so that $\tilde x :=x^*+\theta d_i$ is still feasible (Lecture 5; Slide 11). But then
$$
f(\tilde x)=c^\top \tilde x = c^\top x^*+\theta c^\top d_i=f(x^*)+\theta \bar c_i\le f(x^*),
$$
whence $x^*$ is not the unique optimal solution, the desired contradiction.