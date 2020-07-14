# MAT3007 Assignment 4

[TOC]

## A4.1 

### (a)

In standard form,
$$
\begin{aligned}
&\min\quad &-3x_1-4x_2-3x_3-6x_4&=z\\
&\text{subject to}\quad &2x_1+x_2-x_3+x_4-s_1&=12\\
&& x_1+x_2+x_3+x_4&=8\\
&& -x_2+2x_3+x_4+s_2&=10\\
&& x,s&\ge 0.
\end{aligned}
$$

The auxiliary problem is
$$
\begin{aligned}
&\min\quad &w_1+w_2+w_3&=\hat z\\
&\text{subject to}\quad &2x_1+x_2-x_3+x_4-s_1+w_1&=12\\
&& x_1+x_2+x_3+x_4+w_2&=8\\
&& -x_2+2x_3+x_4+s_2+w_3&=10\\
&& x,s,w&\ge 0.
\end{aligned}
$$
Tableau I:
$$
\begin{array}{|c|ccccccc|c|}
\hline 
& x_1 & x_2 & x_3 & x_4 & s_1 & s_2& w &  \\
\hline

-\hat z_0 & -3 & -1 & -2 & -3 & 1 & -1 & * & -30 \\
-z_0 & -3 & -4 & -3 & -6 & 0 & 0 & * & 0\\
\hdashline
w_1 & \boxed2 & 1 & -1 & 1 & -1 & 0 & * & 12\\
w_2 & 1 & 1 & 1 & 1 & 0 & 0 & * & 8\\
w_3 & 0 & -1 & 2 & 1 & 0 & 1 & * & 10\\
\hline

-\hat z_0 & 0 & 1/2 & -7/2 & -3/2 & -1/2 & -1 & * & -12 \\
-z_0 & 0 & -5/2 & -9/2 & -9/2 & -3/2 & 0 & * & 18\\
\hdashline
x_1 & 1 & 1/2 & -1/2 & 1/2 & -1/2 & 0 & * & 6\\
w_2 & 0 & 1/2 & \boxed{3/2} & 1/2 & 1/2 & 0 & * & 2\\
w_3 & 0 & -1 & 2 & 1 & 0 & 1 & * & 10\\
\hline


-\hat z_0 & 0 & 5/3 & 0 & -1/3 & 2/3 & -1 & * & -22/3 \\
-z_0 & 0 & -1 & 0 & -3 & 0 & 0 & * & 24\\
\hdashline
x_1 & 1 & 2/3 & 0 & 2/3 & -1/3 & 0 & * & 20/3\\
x_3 & 0 & 1/3 & 1 & 1/3 & 1/3 & 0 & * & 4/3\\
w_3 & 0 & -5/3 & 0 & 1/3 & -2/3 & \boxed1 & * & 22/3\\
\hline

-\hat z_0 & 0 & 0 & 0 & 0 & 0 & 0 & * & 0 \\
-z_0 & 0 & -1 & 0 & -3 & 0 & 0 & * & 24\\
\hdashline
x_1 & 1 & 2/3 & 0 & 2/3 & -1/3 & 0 & * & 20/3\\
x_3 & 0 & 1/3 & 1 & 1/3 & 1/3 & 0 & * & 4/3\\
s_2 & 0 & -5/3 & 0 & 1/3 & -2/3 & 1 & * & 22/3\\
\hline

\end{array}
$$
Tableau II:
$$
\begin{array}{|c|cccccc|c|}
\hline 
& x_1 & x_2 & x_3 & x_4 & s_1 & s_2\\
\hline

\hline 
-z_0 & 0 & -1 & 0 & -3 & 0 & 0& 24 \\
\hdashline
x_1 & 1 & 2/3 & 0 & 2/3 & -1/3 & 0 & 20/3\\
x_3 & 0 & 1/3 & 1 & \boxed{1/3} & 1/3 & 0 & 4/3 \\
s_2 & 0 & -5/3 & 0 & 1/3 & -2/3 & 1 & 22/3\\

\hline 
-z_0 & 0 & 2 & 9 & 0 & 3 & 0& 36 \\
\hdashline
x_1 & 1 & 0 & -2 & 0 & -1 & 0 & 4\\
x_4 & 0 & 1 & 3 & 1 & 1 & 0 & 4 \\
s_2 & 0 & -2 & -1 & 0 & -1 & 1 & 6\\
\hline
\end{array}
$$
Hence the original optimal value is $z_\text{max}=-(-36)=36,$ obtained at $x^*=[4;0;0;4].$

### (b)

Note that in the final tableau, all reduced costs of the non-basis variables are strictly positive. Hence the optimal solution is unique.

### (c)

Dual:
$$
\begin{aligned}
&\min\quad &-12y_1+8y_2+10y_3&=z\\
&\text{subject to}\quad &-2y_1+y_2&\ge 3\\
&& -y_1+y_2-y_3&\ge4\\
&& y_1+y_2+2y_3&\ge 3\\
&& -y_1+y_2+y_3&\ge 6\\
&& y_1,y_3&\ge0.
\end{aligned}
$$
To find out the solution of the dual, we first compute the slack for the primal at $x^*:$
$$
\left[\begin{array}{c}
-12\\8\\10
\end{array}\right]-
\left[\begin{array}{cccc}
-2 & -1 & 1 & -1\\
1&1&1&1\\
0&-1&2&1
\end{array}\right]\left[\begin{array}{c}
4\\0\\0\\4
\end{array}\right]=\left[\begin{array}{c}
0\\0\\6
\end{array}\right].
$$
By Complementary Slackness Theorem if $y^*$ is an optimal solution we must have $y^*_{(3)}=0.$ Also we know that the first and forth constraints of the dual must be tight at $y^*.$ We obtain the following linear system:
$$
\begin{aligned}
-2y_1^*+y_2^*=3\\-y_1^*+y_2^*=6
\end{aligned}
$$
yielding a unique optimal dual solution $y^*=[3;9;0].$

### (d)

First consider changing the $c_3$ from $3$ to $3+\lambda.$ Since $x_3$ is not in the optimal basis, for the optimal solution to still be optimal we only need
$$
9-\lambda \ge 0\iff \lambda \le 9.
$$

- If $c_3$ is decreased to $1,$ we have $\lambda=-2\le 9.$ The optimal solution remains unchanged, and since $x_3=0,$ the optimal value is also unchanged.

- If $c_3$ is increased to $12,$ we have $\lambda=9\le 9.$ Again both the optimal solution and value remain unchanged.

Next we consider changing $c_1$ from $3$ to $3+\lambda.$ Since $x_1$ is in the optimal basis $B=\{x_1,x_4,s_2\}$, for the optimal solution to remain optimal we need
$$
\begin{aligned}
(c^*)^\top&= c^\top-c^\top_BA^*\\&=-[3+\lambda,4,3,6,0,0]+[3+\lambda,6,0]
\left[\begin{array}{c}
1 & 0 & -2 & 0 & -1 & 0\\
0 & 1 & 3 & 1 & 1 & 0 \\
0 & -2 & -1 & 0 & -1 & 1 \\
\end{array}\right]\\
&= [0,2,9-2\lambda,0,3-\lambda,0]\ge0\\\\ \iff\lambda&\le 3.
\end{aligned}
$$
In that case, the optimal value becomes
$$
\begin{aligned}
z_0^*&=z_0-c_B^\top b^*\\
&=0+[3+\lambda,6,0][4;4;6]\\
&=36+4\lambda.
\end{aligned}
$$

- If $c_1$ is decreased to $1$, we have $\lambda =-2\le 3.$ Hence optimal solution remains unchanged. However the optimal value becomes $36-8=28.$
- If $c_1$ is increased to $7,$ we have $\lambda=4\gt3.$ Hence both optimal solution and optimal value are expected to change.

### (e)

Consider changing constant $b_2$ from $8$ to $8+\lambda.$ Need only make sure that the optimal solution under current basis is still feasible. That is,
$$
x^*_B=b^*=A_B^{-1}b=\left[\begin{array}{c}
2 & 1 & 0\\
1 & 1 & 0\\
0 & 1 & 1\\
\end{array}\right]^{-1}\left[\begin{array}{c}
12\\8+\lambda\\10
\end{array}\right]=
\left[\begin{array}{c}
4-\lambda\\4+2\lambda\\6-2\lambda
\end{array}\right]\ge0.
$$
Equivalently,
$$
-2\le\lambda\le3.
$$

## A4.2

### (a)

Define $x=[x_1;x_2;x_3]:=[\text{# SR Insur.; # Mortgage Insur.; # LTC Insur.}].$

We wish to
$$
\begin{aligned}
&\max\quad &3300x_1+2000x_2+5000x_3&=z\\
&\text{subject to}\quad &2x_1+x_2+x_3&\le 250\\
&& 3x_1+x_2+2x_3&\le 150\\
&& x_1+2x_2+4x_3&\le 160\\
&& x&\ge0.
\end{aligned}
$$
MATLAB code:

```matlab
cvx_begin
    variable x(3)
    maximize([3300 2000 5000] * x)
    subject to
    [2 1 1; 3 1 2; 1 2 4] * x <= [250; 150; 160]
    x >= zeros(3, 1)
cvx_end
```

yielding
$$
z_\text{max}=257400,\ x^*=[28;0;33].
$$

### (b)

The dual optimal solution is from the reduced costs of the primal slacks, i.e.,
$$
y^*=[0;820;840].
$$
Consider the typical interpretation of the dual in which a buyer wishes to buy all resources (at prices $y_1,y_2,y_3$ for each working hour of three departments) that would entice the company to sell, meanwhile minimizing his own cost. At the optimal solution $y^*$ (a fair shadow price for both sides) notice that
$$
y^*_1+y^*_2+2y^*_3=2500\ge2000=\text{Profit per Mortgage Ins.},
$$
which means the company would make more money by directly selling its labor to the buyer rather than selling any mortgage insurance. Therefore, the company would not sell any mortgage insurance at optimal solution $x^*.$

### (c)

The coefficient matrix
$$
A=\left[\begin{array}{c}
2&1&1&1&0&0\\
3&1&2&0&1&0\\
1&2&4&0&0&1
\end{array}\right].
$$
We consider changing constant $b_3$ from $160$ to $160+\lambda.$ For the current basis $B=\{s_1,x_1,x_3\}$ to remain optimal, we have
$$
x_B^*=b^*=A_B^{-1}b=\left[\begin{array}{c}
161\\28\\33
\end{array}\right]+\left[\begin{array}{c}
1&-7/10&1/10\\
0&2/5&-1/5\\
0&-1/10&3/10
\end{array}\right]\left[\begin{array}{c}
0\\0\\\lambda
\end{array}\right]=\left[\begin{array}{c}
161+\lambda/10\\28-\lambda/5\\33+3\lambda/10
\end{array}\right]\ge0
$$
yielding
$$
-110\le \lambda\le140,
$$
equivalently,
$$
b_3 \in [50,300].
$$

### (d)

Change $c_1$ from $3300$ to $3300+\lambda.$ To keep the current basis optimal we need
$$
\begin{aligned}
(c^*)^\top&=c^\top-c_B^\top A^*\\
&=-[3300+\lambda,2000,5000,0,0,0]+[0,3300+\lambda,5000]\left[\begin{array}{c}
0&1/2&0&1&-7/10&1/10\\
1&0&0&0&2/5&-1/5\\
0&1/2&1&0&-1/10&3/10
\end{array}\right]\\
&=[0,500,0,0,820+2\lambda/5,840-\lambda/5]\ge0\\
\\
\iff &-2050\le\lambda\le 4200,
\end{aligned}
$$
equivalently,
$$
c_1\in[1250,7500].
$$