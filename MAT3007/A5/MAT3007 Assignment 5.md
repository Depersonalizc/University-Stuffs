# MAT3007 Assignment 5

[TOC]

## A5.1 

### (a)

Rewrite
$$
\begin{aligned}
f_\beta(x)&=\frac12(x-b)^\top (x-b)+\frac\beta2\left(\mathbf1^\top x\right)^2\\
&=\frac12\left(x^\top x-2b^\top x+b^\top b\right)+\frac\beta2\left(\mathbf1^\top x\right)^2,
\end{aligned}
$$
where $\mathbf1$ denotes the all-one vector.

Then the gradient is given by
$$
\begin{aligned}

\nabla f_\beta(x)&=\frac12(2x-2b)+\frac{\beta\cdot2}2\left(\mathbf1^\top x\right)\mathbf1\\
&= x-b+\left(\beta\mathbf1^\top x\right)\mathbf1.
\end{aligned}
$$
The Hessian is then
$$
\begin{aligned}
\nabla^2f_\beta(x)&=\left[\begin{array}{c}
1+\beta&\beta&\cdots&\beta\\
\beta & 1+\beta&\cdots&\beta\\
\vdots&\vdots&\ddots&\beta\\
\beta&\beta&\cdots&1+\beta
\end{array}\right]\\
&=\beta\cdot\mathbf1\mathbf1^\top+I.
\end{aligned}
$$

### (b)

Set $\nabla f_\beta(x)=0.$ Then $x_i+\beta\sum_jx_j=b_i$ for all $i.$ In matrix form,
$$
\left[\begin{array}{cccc}
1+\beta&\beta&\cdots&\beta&\\
\beta & 1+\beta&\cdots&\beta&\\
\vdots&\vdots&\ddots&\beta&\\
\beta&\beta&\cdots&1+\beta&
\end{array}\right]x=b
$$
After some gruesome calculation we obtain
$$
\begin{aligned}
x_\beta^* &=\frac1{1+n\beta}\left[\begin{array}{cccc}
1+(n-1)\beta&-\beta&\cdots&-\beta&\\
-\beta & 1+(n-1)\beta&\cdots&-\beta&\\
\vdots&\vdots&\ddots&-\beta&\\
-\beta&-\beta&\cdots&1+(n-1)\beta&
\end{array}\right]b\\
&= \left(I -\frac\beta{1+n\beta}\mathbf1\mathbf1^\top\right)b.
\end{aligned}
$$
To determine whether $x_\beta^*$ is a local minimizer, note that for all $x\neq0,$
$$
\begin{aligned}
x^\top\nabla^2f_\beta(x_\beta^*)x&=
x^\top\left(\beta\mathbf1\mathbf1^\top+I\right)x\\
&=\beta x^\top\mathbf1\mathbf1^\top x+x^\top x\\&=\beta\left(\mathbf1^\top x\right)^2+||x||^2\\&\ge||x||^2\gt0.

\end{aligned}
$$
Thus by SOSC, $x_\beta^*$ is always a local minimizer.

### (c)

We have
$$
x^*=\lim_{\beta\to\infty}x_\beta^*=\lim_{\beta\to\infty}\left(I -\frac\beta{1+n\beta}\mathbf1\mathbf1^\top\right)b=\left(I-\mathbf1\mathbf1^\top\right)b,
$$
and
$$
\mathbf1^\top x^*=\mathbf1^\top\left(I-\mathbf1\mathbf1^\top\right)b=\mathbf1^\top b-\mathbf1^\top\left(\mathbf1\mathbf1^\top\right) b=\mathbf1^\top b-\mathbf1^\top b=0.
$$

### (d)

The set
$$
\left\{\nabla\left(\mathbf1^\top x\right)\right\}=\left\{\mathbf1\right\}
$$
is clearly linearly independent at all feasible points, i.e., LICQ is always satisfied.

Introduce $\mu,$ the dual multiplier for the equality constraint. The Lagrangian is then
$$
\mathcal L(x,\mu)=\frac12||x-b||^2+\mu\cdot\mathbf1^\top x.
$$
Setting
$$
\left.\nabla_x\mathcal L(x,\mu)\right|_{x=x^*}=x^*-b+\mu\cdot\mathbf1=0\implies \mu\cdot\mathbf1=\mathbf1\mathbf1^\top b\implies \mu=\mathbf1^\top b.
$$
By $\text{(c)},$ $x^*$ is always a feasible solution. We have met all KKT conditions at $(x,\mu)=(x^*,\mathbf1^\top b).$ Thus $x^*$ is a KKT point. It is also the only KKT point because if we let
$$
\nabla_x\mathcal L(x,\mu)=0\implies x=b-\mu\cdot\mathbf1,
$$
the Primal Feasibility would impose
$$
\mathbf1^\top x = \mathbf1^\top b - \mu=0\implies \mu=\mathbf1^\top b\implies x=b-\mathbf1^\top b\mathbf1=x^*.
$$
Adding the LICQ, $x^*$ must be the unique local solution.

## A5.2

### (a)

We have
$$
\begin{aligned}
&\min\quad &f(x)&=x_1^2+x_2^2+x_3^2+x_1x_2+x_2x_3-2x_1-5x_2-6x_3\\
&\text{subject to}\quad &g(x)&=x_1+x_2+x_3-1\le0\\
&& h(x)&=x_1-x_2^2=0.
\end{aligned}
$$
The Lagrangian for the problem is
$$
\mathcal L(x,\lambda,\mu)=f(x)+\lambda g(x)+\mu h(x).
$$
The KKT conditions are:

- Main Condition
  $$
  \nabla_x \mathcal L(x,\lambda,\mu)=\left[\begin{array}{c}
  2x_1+x_2-2+\lambda+\mu\\
  2x_2+x_1+x_3-5+\lambda-2\mu x_2\\
  2x_3+x_2-6+\lambda\end{array}\right]=0.
  $$

- Dual feasibility
  $$
  \lambda\ge0.
  $$

- Primal feasibility
  $$
  g(x)\le0,\ h(x)=0.
  $$

- Complementarity
  $$
  \lambda g(x)=0.
  $$

### (b)

Clearly $x^*$ is a feasible point. Set $\nabla_x\mathcal L(x,\lambda,\mu)|_{x=x^*}=0.$ We have
$$
[\lambda+\mu-2;\lambda-4;\lambda-4]=0\implies \lambda=4,\ \mu=-2.
$$
So the dual feasibility is met. Also, $g(x^*)=1-1=0,$ meeting the complementarity condition. Thus $x^*$ is a KKT point.

We then compute the Hessian at $x^*$
$$
H:=\nabla_{xx}^2\mathcal L(x^*,4,-2)=\left[\begin{array}{c}
2&1&0\\
1&6&1\\
0&1&2\end{array}\right]
$$
We see that the determinant of all leading principals of $H:$
$$
\Delta_1=2,\ \Delta_2=11,\ \Delta_3=20,
$$
are positive. Thus $H\succ0,$ whence the SOSC holds and $x^*$ is a strict local minimizer.

## A5.3

### (a)

<img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\A5\desmos-graph.png" alt="desmos-graph" style="zoom: 50%;" />

### (b)

At $\bar x=[0;1],$ we have
$$
g_1(\bar x)=1-1=0,\ g_2(\bar x)=1-1=0.
$$
Thus both constraints are active, i.e., $\mathcal A(\bar x)=\{1,2\}.$

The gradients of the inequality constraints are
$$
\nabla g_1(x)=[2x_1;2x_2],\ \nabla g_2(x)=[2x_1-2;-2x_2].
$$
At $\bar x,$ the gradients are
$$
\nabla g_1(\bar x)=[0;1],\ \nabla g_2(x)=[-2;-2].
$$
The linearized tangent set at $\bar x$ is then given by
$$
\mathcal T_\ell(\bar x)=\left\{
[d_1;d_2]:d_2\le0,\ d_1+d_2\ge0
\right\}.
$$
We plot the set **after** **shifting the start of all direction vectors to** $\bar x:$

<img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\A5\desmos-graph (1).png" alt="desmos-graph (1)" style="zoom: 50%;" />

### (c)

Since the objective function $f$ is continuous and the feasible set $\Omega$ is compact, by the Extreme Value Theorem $f(\Omega)$ is also compact. Hence $f$ must attain the minimum at some $x^*\in\Omega.$

### (d)

The Lagrangian:
$$
\mathcal L(x,\lambda)=x_2^2-2x_1+\lambda_1(x_1^2+x_2^2-1)+\lambda_2(x_1^2-x_2^2-2x_1+1).
$$
Main condition:
$$
\nabla_x\mathcal L(x,\lambda)=[-2+2\lambda_1x_1+2\lambda_2x_1-2\lambda_2;2x_2+2\lambda_1x_2-2\lambda_2x_2]=0.
$$
Dual feasibility:
$$
\lambda\ge0.
$$
Complementarity:
$$
\lambda_1(x_1^2+x_2^2-1)=0,\ \lambda_2(x_1^2-x_2^2-2x_2+1)=0.
$$
yielding the KKT point
$$
\boxed{\begin{aligned}
x^*&=[1;0],\ \lambda=[1;t],\ t\ge0.
\end{aligned}}
$$
Compute the Hessian at $x^*$:
$$
H:=\nabla_{xx}\mathcal L(x^*,\lambda)=2\left[\begin{array}{c}
1+t&0\\
0&2-t\end{array}\right].
$$
We may choose $t=0\implies H\succ0.$ By SOSC $x^*$ is a strict local minimizer with $f(k_2)=-2.$ But since all feasible points other than $x^*$ are regular, we must attain global minimum at $x^*,$ with optimal value $f(x^*)=-2.$