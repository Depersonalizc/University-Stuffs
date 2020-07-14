# MAT3007 Assignment 3

[TOC]

## A3.1 

### (a)

$$
\begin{aligned}
&\min\quad &4y_1+7y_2\\
&\text{subject to}\quad &2y_1+y_2&\ge5\\
&& 3y_1+2y_2&\ge2\\
&& y_1+3y_2&\ge5\\
&& y_1,y_2&\ge 0.
\end{aligned}
$$

### (b)

<img src="C:\Users\chen1\AppData\Roaming\Typora\typora-user-images\image-20200704032514177.png" alt=" " style="zoom: 33%;" />

The unique optimal solution is $\color{purple}y^*=[y_1;y_2]=[2;1],$ with optimal value $\color{blue}15.$

### (c)

Let $x^*$ denote the optimal solution for the primal. Using the Complementary Slackness Theorem and the fact that $y^*\gt0,$ the constraints in the dual must both be tight. Also, computing the slacks for the dual,
$$
v=[5-5;8-2;5-5]=[0;6;0],
$$
by the same theorem we know that $(x^*)_2$ must be $0.$ 

These information transform the primal into a linear system of equation:
$$
2x_1+x_3=4\\x_1+3x_3=7
$$
which yields $x_1=1,x_3=2.$ 

Hence the optimal solution for the primal is $x^*=[1;0;2]$ with optimal value $15,$ coinciding with that of the dual.

## A3.2

Rewriting the LP,
$$
\min_x\ c^\top x\quad \text{s.t.}\quad 
\left[\begin{array}{c}
-A\\C
\end{array}\right]x\ 
\begin{array}{c}
\ge\\=
\end{array}\left[
\begin{array}{c}
b\\d
\end{array}\right].
\tag{2'}
$$
Its dual is then given by
$$
\max_{y\in \R^m,z\in\R^p}\ [b^\top,d^\top] y\quad \text{s.t.}\quad [-A^\top,C^\top]y= c,\ y\ge0.
$$
The dual of the dual is then
$$
\min_x\ c^\top x\quad \text{s.t.}\quad 
\left[\begin{array}{c}
-A\\C
\end{array}\right]x\ 
\begin{array}{c}
\ge\\=
\end{array}\left[
\begin{array}{c}
b\\d
\end{array}\right].
$$
which is equivalent to $ \text{(2')},$ hence to $(2).$

## A3.3

### (a)

The Strong Duality Theorem says there is no such an example.

### (b)

Primal:
$$
\min_{x\in \R}\ 0^\top x\quad \text{s.t.}\quad 0x=0.
$$
Dual:
$$
\max_{y\in \R}\ 0^\top y\quad \text{s.t.}\quad 0^\top y=0.
$$

### (c)

Primal:
$$
\min_{x\in \R}\ 0^\top x\quad \text{s.t.}\quad 1x=0.
$$
Dual:
$$
\max_{y\in \R}\ 0^\top y\quad \text{s.t.}\quad 1^\top y=0.
$$

### (d)

Primal:
$$
\min_{x\in \R^2}\ 0^\top x\quad \text{s.t.}\quad [1,-1]x=0,\ x\ge0.
$$
Dual:
$$
\max_{y\in \R}\ 0^\top y\quad \text{s.t.}\quad [1;-1] y\le[0;0].
$$
It can be easily checked that $x^*=[0;0]$ is a degenerate optimal BFS for the primal, and that $y^*=0$ is the unique optimal solution for the dual.

## A3.4

### (a)

Let $a_i$ denotes the $i$-th column of $A.$ Then
$$
a_i^\top x=\sum_{j=1}^4a_{ji}x_j=\mathbb E[\text{player I's winning}|\text{player II chooses } i ],\quad i=1:4.
$$
Since $t$ is a lower bound for $a_i^\top x,$ finding the max of $t$ is equivalent to maximizing the minimum of $a_i^\top x,$ i.e., finding the optimal probabilistic strategy $x$ for player I in the sense of maximizing his expected winning in the worst case.

MATLAB code:

```matlab
cvx_begin
    variables x(4) t
    maximize(t)
    subject to
    A' * x >= t * ones(4, 1)
    ones(1, 4) * x == 1
    x >= zeros(4, 1)
cvx_end
```

yielding
$$
p^*=t_{\max}=0,
$$
obtained at
$$
x^*=[0.088;0.338;0.412;0.162],\ t^*=0.
$$

### (b)

Rewrite $\text{(3)}$,
$$
\begin{aligned}
&\max_{x,t}\quad &[0_{1\times 4},1][x;t]\\
&\text{subject to}\quad &[-A^\top,1_{4\times1}][x;t]&\le0_{4\times 1}\\
&& [1_{1\times4},0][x;t]&=1\\
&& x&\ge0.
\end{aligned}
$$
Its dual is then given by
$$
\begin{aligned}
&\min_{y,s}\quad &[0_{1\times 4},1][y;s]\\
&\text{subject to}\quad &[-A,1_{4\times1}][y;s]&\ge0_{4\times 1}\\
&& [1_{1\times 4},0][y;s]&=1\\
&& y&\ge0,
\end{aligned}
$$
or equivalently
$$
\begin{aligned}
&\min_{y,s}\quad &s\\
&\text{subject to}\quad &Ay&\le s\cdot1\\
&& 1^\top y&=1\\
&& y&\ge0.
\end{aligned}
$$
This time we wish to minimize the upper bound $s$ for $A_i^\top y,\ i=1:4,$ where $A_i^\top$ is the $i$-th row of $A.$ If we interpret $y$ as the probabilistic strategy for player II, then
$$
A_i^\top y=\sum_{j=1}^4a_{ij}y_i=E[\text{player II's losses}|\text{player I chooses } i ],\quad i=1:4.
$$
We see that the dual is to find the optimal probabilistic strategy $y$ for player II in the sense of minimizing his expected loss in the worst case.

MATLAB code:

```matlab
cvx_begin
    variables y(4) s
    maximize(s)
    subject to
    A * y <= s * ones(4, 1)
    ones(1, 4) * x == 1
    y >= zeros(4, 1)
cvx_end
```

which yields
$$
d^*=s_{\min}=0,
$$
at
$$
y^*=[0.25;0.50;0.25;0.00],\ s^*=0.
$$

### (c)

First note that
$$
\begin{aligned}
\max_t \ t\quad \text{s.t.}\quad Ax\ge t\cdot1&=\max_t \ t\quad \text{s.t.}\quad \min_{i}\ A_i^\top x\ge t\\
&=\min_{i}\ A_i^\top x,
\end{aligned}
$$
where $A_i^\top$ denotes the $i$-th row of $A,$ $i=1:4.$ 

Now let $ m:=\text{argmin}_{i}\ A_i^\top x.$ We have
$$
\begin{aligned}
\min_{y\in P}\ y^\top Ax &=\min_{y\in P}\ \sum_i y_i A_i^\top x\\
&= \min_{y\in P}\ \left\{y_mA_m^\top x+\sum_{i\neq m}y_iA_i^\top x\right\}.
\end{aligned}
$$
We claim that this minimum is exactly $A_m^\top x,$ obtained at $\hat y,$ the all-zero vector except $\hat y_m=1.$ Indeed, for any $y\in P,$ we have
$$
\begin{aligned}
y^\top Ax - \hat y^\top A x&=y^\top Ax- A_m^\top x\\&=(y_m-1)A_m^\top x+\sum_{i\neq m}y_iA_i^\top x\\&\ge
(y_m-1)A_m^\top x+\sum_{i\neq m}y_iA_m^\top x\\&=\left(-1+\sum_i y_i\right)A_m^\top x\\&=0.

\end{aligned}
$$
Thus
$$
\begin{aligned}
&\quad\quad\ \ \max_{Ax\ge t\cdot 1}\ t=\min_i\ A_i^\top x=A_m^\top x=\min_{y\in P} \ y^\top A x\\
&\implies \max_{x\in P} \max_{Ax\ge t\cdot 1}\ t=p^*= \max_{x\in P} \min_{y\in P} \ y^\top Ax.
\end{aligned}
$$
Finally by Strong Duality,
$$
p^*=\max_{x\in P} \min_{y\in P} \ y^\top Ax=d^*.
$$

### (d)

The game is fair in the sense that the expected winning for player I (or expected loss for player II) is zero in the worst case. Using only numbers one and two, the pay-off matrix becomes
$$
B = \left[\begin{array}{cc}
-2&3\\3&-4

\end{array}\right].
$$
Substituting matrix $A$ with $B$ in $(3)$ and solving the problem in MATLAB with the code below,

```matlab
cvx_begin
    variables x(2) t
    maximize(t)
    subject to
    B' * x >= t * ones(2, 1)
    ones(1, 2) * x == 1
    x >= zeros(2, 1)
cvx_end
```

we obtain $p^*=t_{\max}=0.0833>0$ with strategy $x^*=[0.5833;0.4167],$ which indicates a preference of the new game towards player I. 

## A3.5

### (a)

$$
\begin{aligned}
&\min_{x,t}\quad &t\\
&\text{subject to}\quad &t\cdot 1_{m\times1}&\ge Ax-b\\
&& t\cdot 1_{m\times1}&\ge b-Ax\\
&& t&\ge0.
\end{aligned}\tag {4'}
$$

### (b)

Rewrite $(4'),$
$$
\begin{aligned}
&\min_{x,t}\quad &[0_{1\times n},1][x;t]\\
&\text{subject to}\quad &
\left[\begin{array}{c|c}
-A&1_{m\times1}\\
\hline
A& 1_{m\times 1}
\end{array}\right]
\left[\begin{array}{c}
x\\t
\end{array}\right]
&\ge \left[\begin{array}{c}
-b\\b
\end{array}\right]\\&& t&\ge 0.
\end{aligned}
$$
Thus the dual is given by
$$
\begin{aligned}
&\max_{z\in\R^{2m}}\quad &[-b^\top,b^\top]z\\
&\text{subject to}\quad &
\left[\begin{array}{c|c}
-A^\top&A^\top\\
\hline
1_{1\times m}& 1_{1\times m}
\end{array}\right]
z\ &\begin{array}{c}
=\\\le
\end{array}
\left[\begin{array}{c}
0_{n\times1}\\1
\end{array}\right]\\&&z&\ge0.
\end{aligned}\tag{4'D}
$$

### (c)

If we denote $z^-:=z[1:m]$ and $z^+:=z[m+1:2m],$ we have
$$
\begin{aligned}
&\max_{z^+,z^-\ge 0}\quad &b^\top(z^+-z^-)\\
&\text{subject to}\quad &
A^\top(z^+-z^-)&=0
\\&&1^\top (z^+ + z^-)&\le 1.
\end{aligned}
$$
Setting $y:= z^+-z^-,$ the problem further transforms into
$$
\begin{aligned}
&\max_{z^+,z^-\ge0,y}\quad &b^\top y\\
&\text{subject to}\quad  &y&=z^+-z^-\\&&
A^\top y&=0
\\&&1^\top (z^+ + z^-)&\le 1.
\end{aligned}\tag{5}
$$
Since the objective functions coincide now (and only depend on $y$), to prove the equivalence, it remains to show that ranges of $y$ are equal in two problems:
$$
||y||_1\le 1\iff \begin{aligned}
&y=z^+-z^-\\
&1^\top(z^++z^-)\le 1\\
&z^+,z^-\ge0.

\end{aligned}
$$
Suppose we have $||y||_1\le1,$ then we may define, for all $i,$
$$
z_i^+:=\begin{cases}
y_i, \quad\text{if }y_i\ge0,
\\0,\quad\text{otherwise};
\end{cases}\quad 
z_i^-:=\begin{cases}
-y_i, \quad\text{if }y_i\lt 0,
\\0,\quad\text{otherwise}.
\end{cases}
$$
Clearly $z^+,z^-\ge0$, and $z^+-z^-=y.$ Also,
$$
1^\top(z^++z^-)=\sum_{i=1}^mz_i^++z_i^-=\sum_{i=1}^m|y_i|=||y||_1\le 1.
$$
Conversely, suppose there exists some $(y,z^+,z^-)$ rendering RHS true. We have
$$
||y||_1=\sum_{i=1}^m|y_i|=\sum_{i=1}^m|z_i^+-z_i^-|\le\sum_{i=1}^m|z_i^+|+|z_i^-|=\sum_{i=1}^mz_i^++z_i^-=1^\top(z^++z^-)\le 1,
$$
which completes the proof.

### (d)

We already have $\text{RHS}\equiv(5)\equiv(4'\text D).$ To show that the optimal value equals that of $ \text{LHS}\equiv(4)\equiv(4'),$ it suffices to show that $(4')$ and its dual $(4'\text D)$ have the same optimal value. 

First note that both problems are feasible: For $(4'), (x,t)=(0,||b||_\infty)$ is a feasible solution; for $(4'\text D),z=0$ is feasible. Now $(4')$ must be bounded since otherwise $(4'\text D)$ would not be feasible due to the duality gap. Therefore $(4')$ attains a finite optimal value $m,$ at some point $p^*.$ By the Strong Duality, $(4'\text D)$ must have $p^*$ as its optimal value as well, and we are done.

### (e)

MATLAB code:

```matlab
m = 100; 
A = [ones(m), ones(m)];
b = (1:m)';

% original problem
tic;
cvx_begin quiet
    variable x(2 * m)
    minimize(norm(A * x - b, inf))
cvx_end
toc
Elapsed time is 0.441330 seconds.

% dual problem
tic;
cvx_begin quiet
    variable y(m)
    maximize(b' * y)
    subject to
    A' * y == zeros(2 * m, 1)
    norm(y, 1) <= 1
cvx_end
toc
Elapsed time is 0.249609 seconds.
```

Both methods obtain the optimal value $49.5,$ but solving dual is nearly twice as fast.