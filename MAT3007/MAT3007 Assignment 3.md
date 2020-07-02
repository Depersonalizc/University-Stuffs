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

<img src="C:\Users\Jamie\Desktop\desmos-graph.png" alt="desmos-graph" style="zoom: 50%;" />

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
\min_x\ c^\top x\quad \text{s.t.}\quad -Ax\ge b,\ Cx=d,\ x \text{ free.}\tag{2'}
$$
Its dual is then given by
$$
\max_y\ [b;d]^\top y\quad \text{s.t.}\quad [-A;C]^\top y= c,\ y_{1:m}\ge0,\ y_{m+1:m+p}\text{ free.}
$$
The dual of the dual is then
$$
\min_x\ c^\top x\quad \text{s.t.}\quad -Ax\ge b,\ Cx=d,\ x\text{ free,}
$$
which is equivalent to $\text{(2')},$ hence to $(2).$

## A3.3

### (a)

The Duality Theorem says there is no such an example.

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
\min_{x\in \R}\ 0^\top x\quad \text{s.t.}\quad 1x=1.
$$
Dual:
$$
\max_{y\in \R}\ 1^\top y\quad \text{s.t.}\quad 1^\top y=0.
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
We can easily check that $x^*=[0;0]$ is a degenerate optimal BFS for the primal, and that $y^*=0$ is the unique optimal solution for the dual.

## A3.4

### (a)

Let $a_i$ denotes the $i$-th column of $A.$ Then
$$
a_i^\top x=\sum_{j=1}^4a_{ji}x_j=\mathbb E[\text{player I's winning}|\text{player II chooses } i ],\quad i=1,\cdots,4.
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
p^*=0,
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
This time we wish to minimize the upper bound $s$ for $A_i^\top y,\ i=1,\cdots,4,$ where $A_i^\top$ is the $i$-th row of $A.$ If we interpret $y$ as the probabilistic strategy for player II, then
$$
A_i^\top y=\sum_{j=1}^4a_{ij}y_i=E[\text{player II's losses}|\text{player I chooses } i ],\quad i=1,\cdots,4.
$$
We see the dual is to find the optimal probabilistic strategy $y$ for player II in the sense of minimizing his expected loss in the worst case.

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
d^*=0,
$$
at
$$
y^*=[0.25;0.50;0.25;0.00],\ s^*=0.
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