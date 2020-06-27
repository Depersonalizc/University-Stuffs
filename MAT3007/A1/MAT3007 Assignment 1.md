# MAT3007 Assignment 1

[TOC]

## A1.1 

### (a)

Define $n:= \text{# floors aboveground}, m:= \text{# floors underground},$ and $u:= \text{uniform height of floors}.$ So $h=un,d=um.$

The optimization formulation is as follows:
$$
\begin{aligned}
&\min_{l,w,h,d,n,m,u}\quad &lwd\\
&\text{subject to}\quad &w\le l\le 2w\\
&& un=h\\
&& um = d\\
&& u\ge 7/2\\
&&l\le40\\
&&l \le h\\
&& 1/10\le m/({n+m}) \le 1/4\\
&& lw(n+m) \ge 10000\\
&& lw+2h(l+w) \le 5000\\
&& l,w,h,u\in \R^+\\
&& n,m\in \N^+.
\end{aligned}
$$

### (b)

A feasible point:
$$
\left[\begin{array}{c}
l & w & h & d &n & m&u
\end{array}\right] =
\left[\begin{array}{c}
27 & 27 & 38.5 & 10.5 &11 & 3&3.5
\end{array}\right].
$$

## A1.2

### (a)

Let $x_1,x_2$ be the $\text{# product I,II}$ respectively. Then the LP formulation is:
$$
\begin{aligned}
&\max_{x_1,x_2}\quad &8x_1+7x_2\\
&\text{subject to}\quad &x_1/3+x_2/4\le100\\
&& x_1/5+x_2/4\le70\\
&& x_1,x_2\ge0.
\end{aligned}
$$

### (b)

$$
\begin{aligned}
&\min_{x_1,x_2,s_1,s_2}\quad &-8x_1-7x_2\\
&\text{subject to}\quad &x_1/3+x_2/4+s_1=100\\
&& x_1/5+x_2/4+s_2=70\\
&& x_1,x_2,s_1,s_2\ge 0,
\end{aligned}
$$

or equivalently in matrix form,
$$
\begin{aligned}

&\min_{x=(x_1,x_2,s_1,s_2)^\top}\quad &
\left[\begin{array}{c}
-8&-7&0&0
\end{array}\right]
x\\
&\text{subject to}\quad &
\left[\begin{array}{c}
1/3&1/4&1&0\\
1/5&1/4&0&1\\
\end{array}\right]x=
\left[\begin{array}{c}
100\\
70
\end{array}\right]
\\
&& x\ge 0.
\end{aligned}
$$

### (c)

Define $x_3:= \text{# overtime assembly hours}. $ LP formulation:
$$
\begin{aligned}
&\max_{x_1,x_2,x_3}\quad &8x_1+7x_2-7x_3\\
&\text{subject to}\quad &x_1/3+x_2/4\le100+x_3\\
&& x_1/5+x_2/4\le70\\
&& x_3 \le 60\\
&& x_1,x_2,x_3\ge0.
\end{aligned}
$$

### (d)

```matlab
>> cvx_begin
    variable x
    variable y
    maximize (8 * x + 7 * y)
    subject to
        x/3 + y/4 <= 100
        x/5 + y/4 <= 70
        x >= 0
        y >= 0
cvx_end
```

yielding
$$
\text{max_profit}(x^*,y^*) = 2500
$$
with
$$
\left[\begin{array}{c}
x^* & y^*
\end{array}\right]
=
\left[\begin{array}{c}
225 & 100
\end{array}\right]
$$

## A1.3

### (a)

Let $A=[a_{ij}]\ge0$ be the actual flow and $C=[c_{ij}]\ge0$ the capacity, both from $v_i$ to $v_j.$ Then the LP formulation is:

$$
\begin{aligned}
&\max_{A}\quad &\sum_{i} a_{in}&\\
&\text{subject to}
\quad &0\le A\le C&\\
&& \sum_j a_{ji}=\sum_j a_{ij}&&\forall i,j\in\{1,...,n\} 
\end{aligned}
$$
*Note that we define $c_{ij}:=0$ if $(i,j)\not\in E.$

### (b)

```matlab
% capacity values
C = [0    11     8     0     0     0
     0     0    10    12     0     0
     0     1     0     0    11     0
     0     0     4     0     0    15
     0     0     0     7     0     4
     0     0     0     0     0     0];

cvx_begin
  variable A(6,6)
  maximize (ones(1,6) * A(:,6)) % 6th col sum
  subject to
    for i = 1 : 6*6
      % flow capacity restriction
      0 <= A(i) <= C(i)
    end
    for i = 2 : 6-1
      % flow conservation, ith row sum == ith col sum
      A(i,:) * ones(6,1) == ones(1,6) * A(:,i) 
    end
cvx_end
```

yielding
$$
\text{max_flow}(A^*) = 19
$$
with
$$
A^*=\left[\begin{array}{c}
0&11&8&0&0&0\\
0&0&1.5947&10.2690&0&0\\
0&0.8637&0&0&9.3653&0\\
0&0&0.6343&0&0&15\\
0&0&0&5.3653&0&4\\
0&0&0&0&0&0
\end{array}\right].
$$

## A1.4

### (a)

$$
\min_{x\in\R^n}\ c^{\top}x\quad \text{s.t.}\quad -\delta \le (Ax-b)_i\le \delta\quad\forall i.
$$

To show equivalence, it suffices to show equality of two feasible regions (as objective functions are the same).

Say we have $x\in \R^n$ with $-\delta \le (Ax-b)_i\le \delta,\ x\ge0 \quad \forall i\in\{1,\cdots,m\}.$ Then
$$
\begin{aligned}
&\quad&-\delta \le (Ax-b)_i\le \delta,\ x\ge0\quad \forall i\in\{1,\cdots,m\}\\
&\implies&|(Ax-b)_i|\le \delta,x\ge0\quad \forall i\in\{1,\cdots,m\}\\
&\implies& \max_{1\le i\le m}|(Ax-b)_i| \le \delta,x\ge0\\
&\implies& ||Ax-b||_\infty \le \delta,x\ge0.

 \end{aligned}
$$
Conversely, suppose $||Ax-b||_\infty \le \delta,x\ge0.$ We have
$$
\begin{aligned}
&\quad&||Ax-b||_\infty \le \delta,x\ge0\\
&\implies&\max_{1\le i\le m}|(Ax-b)_i| \le \delta,x\ge0\\
&\implies& |(Ax-b)_j|\le\max_{1\le i\le m}|(Ax-b)_i|\le \delta,x\ge0\quad \forall j\in\{1,\cdots,m\}\\
&\implies& -\delta \le (Ax-b)_i\le \delta,\ x\ge0\quad \forall i\in\{1,\cdots,m\}.

 \end{aligned}
$$
Therefore two feasible regions are indeed equal, hence the equivalence.

### (b)

Let $x=\left[\begin{array}{c}
x_1 & x_2
\end{array}\right]^\top:=\left[\begin{array}{c}
\text{# salad A}&\text{# salad B}
\end{array}\right]^\top.$ The LP can be formulated in the standard form as follows:
$$
\begin{aligned}
&\min_{x}\quad &\left[\begin{array}{c}
-10&-20
\end{array}\right]x\\
&\text{subject to}&

\left[\begin{array}{c}
1/4&1/2\\
1/8&1/4\\
5&1\\
\end{array}\right]x=
\left[\begin{array}{c}
25\\
10\\
120\\
\end{array}\right]\\
&& x\ge0.
\end{aligned}
$$

### (c)

Suppose $\hat x$ is a solution to the constraint. Then
$$
25 = \left[\begin{array}{c}
1/4&1/2
\end{array}\right] \hat x =2\left[\begin{array}{c}
1/8&1/4
\end{array}\right]\hat x=2\cdot10=20,
$$
a contradiction. Hence the LP is infeasible.

### (d)

From (a) we have the equivalent formulation of the robust LP:
$$
\begin{aligned}
&\max_{x}\quad &10x_1+20x_2\\
&\text{subject to}&x\ge0\\&&
\color{red}\text{(mango)}\quad-5\le x_1/4+x_2/2-25\le 5\\
&&\color{green}\text{(pineapple)}\quad-5\le x_1/8+x_2/4-10\le 5
\\&&\color{purple}\text{(strawberry)}\quad-5\le x_1/8+x_2/4-10\le 5

\end{aligned}
$$
The following graph shows the three constraints separately given $x\ge0$ (horizontal: $x_1$, vertical: $x_2$).

<img src="C:\Users\chen1\AppData\Roaming\Typora\typora-user-images\image-20200615032619194.png" alt="image-20200615032619194" style="zoom:50%;" />

The graph below shows the feasible set in red (overlap of three regions above).

<img src="C:\Users\chen1\AppData\Roaming\Typora\typora-user-images\image-20200615031933067.png" alt="image-20200615031933067" style="zoom:40%;" />

where the black dotted line is the contour of the profit function at maximum, $10x_1+20x_2=1200.$ We see there's an infinite number of solutions (the entire upper edge of the feasible region) yielding the optimal profit ${1200\text{ RMB}}$ with active constraints
$$
x_1/4+x_2/2-25\le 5,\\
x_1/8+x_2/4- 10\le5.
$$

Nevertheless, $(x_1^*,x_2^*)=(14,53)$ is the only lattice point among the solutions and will be our final production plan. Following this plan, the amounts of fruits used are $\text{# mango} = x_1^*/4+x_2^*/2=30$; $\text{# pineapple} = x_1^*/8+x_2^*/4=15$; $\text{# strawberry} = 5x_1^*+x_2^*=123$.

