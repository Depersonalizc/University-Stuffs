# Assignment 4

## Table of contents

[TOC]







## 1

### (a)

$\forall A \in M_{n\times n},$
$$
A = I_n^{-1}AI_n=A.
$$

### (b)

Suppose $\exists T  $ s.t.
$$
B = T^{-1}AT.
$$
Take $S := T^{-1}. $ Then $S^{-1}=T,$
$$
A=S^{-1}BS.
$$

### (c)

Suppose $\exists P, Q $ s.t.
$$
A = P^{-1}BP\\
B = Q^{-1}CQ
$$
Take $S:=QP.$ Then $S^{-1}=P^{-1}Q^{-1},$
$$
A = S^{-1}CS.
$$

## 2

The matrix representation of $L$ w.r.t. standard basis $\Epsilon _2$
$$
\text{Rep}_{\Epsilon_2}(L) = 
\left[
\begin{array}{cc}
3& 0\\
1& -1
\end{array}
\right]_{\Epsilon_2}.\label4
$$
Change-of-basis matrix from $B$ to $\Epsilon_2$
$$
P:=P_{B,\Epsilon_2} =\left[
\begin{array}{cc}
1& 2\\
2& 3
\end{array}
\right]_{B,\Epsilon_2}.\label3
$$
Then the change-of-basis matrix from $\Epsilon_2$ to $B$ is simply the inverse
$$
P^{-1}=P_{\Epsilon_2,B}=\left[
\begin{array}{cc}
-3& 2\\
2& -1
\end{array}
\right]_{\Epsilon_2,B}.
$$
It follows that
$$
\begin{aligned}
\text{Rep}_{B}(L)&=P^{-1}\text{Rep}_{\Epsilon_2}(L)P\\&=
\left[
\begin{array}{cc}
-3& 2\\
2& -1
\end{array}
\right]_{\Epsilon_2,B}
\left[
\begin{array}{cc}
3& 0\\
1& -1
\end{array}
\right]_{\Epsilon_2}\left[
\begin{array}{cc}
1& 2\\
2& 3
\end{array}
\right]_{B,\Epsilon_2}
\\&=
\left[
\begin{array}{cc}
-11& -20\\
7& 13
\end{array}
\right]_{B}.
\end{aligned}
$$

## 3

### (a)

No. 

Given a polynomial $f$ of degree $\geq 3, 0\cdot f = 0$ is of degree $0<3.$

### (b)

No.

Given a polynomial $f$ s.t. $f(1)+2f(2)=1,$ $g:=0\cdot f=0$ is constant zero, implying $g(1)+2g(2)=0\neq 1.$

### (c)

Yes.

Given polynomials $f_1,f_2$ with $f(x)=f(1-x),$ let $g:=\alpha f_1+\beta f_2.$ Then $g(x) = \alpha f_1(x)+\beta f_2(x)=\alpha f_1(1-x)+\beta f_2(1-x)=g(1-x).$

## 4

### (a)

Either a **parallelogram**, or a **straight line**, or a single **point**.

### (b)

The region remains square after transformation $A$ if and only if
$$
A^TA=AA^T=cI_2, \ c \neq 0.
$$

## 5

Denote ordered bases $\Epsilon_3:=\{e_1, e_2, e_3\}, B := \{b_1,b_2\}.$

The linear map
$$
L:\R^3\to\R^2
$$
is characterized by its action on the basis $\Epsilon_3$:
$$
\Epsilon_3=\{e_1, e_2, e_3\}\xrightarrow{L}\{L(e_1),L(e_2),L(e_3)\}\xrightarrow{P_{\Epsilon_2,B}}\text{Rep}_B\{L(e_1), L(e_2), L(e_3)\}
$$
where $P_{\Epsilon_2,B}$ is the change of basis matrix from $\Epsilon_2$ to $B$,
$$
P_{\Epsilon_2,B}=P^{-1}_{B,\Epsilon_2}=
\left[
\begin{array}{c}
b_1,b_2
\end{array}
\right]^{-1}=\frac12\left[
\begin{array}{cc}
1 &1\\
-1& 1
\end{array}
\right].
$$
Hence
$$
\begin{aligned}A = \text{Rep}_{\Epsilon_3,B}(L)&=P_{\Epsilon_2,B}[L(e_1),L(e_2),L(e_3)]\\
&=\frac12\left[
\begin{array}{cc}
1 &1\\
-1& 1
\end{array}
\right]

\left[
\begin{array}{ccc}
1 &-1&-1\\
1& 1&1
\end{array}
\right]\\&=
\left[
\begin{array}{ccc}
1 &0&0\\
0& 1&1
\end{array}
\right]_{\Epsilon_3,B}.
\end{aligned}
$$

## 6

Denote ordered bases $B=\{b_1,b_2, b_3\}, U = \{u_1,u_2\}.$

The linear map
$$
L:\R^2\to\R^3
$$
is characterized by its action on the basis $U$:
$$
U=\{u_1, u_2\}\xrightarrow{L}\{L(u_1),L(u_2)\}\xrightarrow{P_{\Epsilon_3,B}}\text{Rep}_B\{L(u_1), L(u_2)\}
$$
where $P_{\Epsilon_3,B}$ is the change-of-basis matrix from $\Epsilon_3$ to $B$,
$$
P_{\Epsilon_3,B}=P^{-1}_{B,\Epsilon_3}=
\left[
\begin{array}{c}
b_1,b_2,b_3
\end{array}
\right]^{-1}=\left[
\begin{array}{ccc}
1 &-1 &0\\
0& 1 &-1\\
0 &0 &1
\end{array}
\right].
$$
Hence
$$
\begin{aligned}\text{Rep}_{B,U}(L)&=P_{\Epsilon_3,B}[L(u_1),L(u_2)]\\
&=\left[
\begin{array}{ccc}
1 &-1 &0\\
0& 1 &-1\\
0 &0 &1
\end{array}
\right]

\left[
\begin{array}{cc}
2 &1\\
3& 4\\
-1&2
\end{array}
\right]\\&=
\left[
\begin{array}{cc}
-1 &-3\\
4& 2\\
-1& 2
\end{array}
\right]_{B,U}.
\end{aligned}
$$

## 7

The linear map
$$
T:\Bbb P^2\to\Bbb P^2
$$
is characterized by its action on basis $\epsilon_2$:
$$
\epsilon_2=\{1, x, x^2\}\xrightarrow{T}\{1,3x-2,9x^2-12x+4\}\xrightarrow{\text{Rep}_{\epsilon_2}}\left\{
\left[
\begin{array}{c}
1\\
0\\
0
\end{array}
\right],\left[
\begin{array}{c}
-2\\
3\\
0
\end{array}
\right],\left[
\begin{array}{c}
4\\
-12\\
9
\end{array}
\right]
\right\}_{\epsilon_2}.
$$
Hence
$$
\text{Rep}_{\epsilon_2}(T)=\left[
\begin{array}{ccc}
1 &-2 &4\\
0& 3 &-12\\
0 &0 &9
\end{array}
\right]_{\epsilon_2}.
$$

## 8

Denote the basis $J:=\{v_1,v_2,v_3\}.$

The change-of-basis matrix from $J$ to $\Epsilon_3$
$$
V =[v_1,v_2,v_3]=\left[
\begin{array}{ccc}
1&1&0\\
1&2&-2\\
1&0&1
\end{array}
\right]_{J,\Epsilon_3}.
$$
Note that the change-of-basis matrix from $\Epsilon_3$ to $J$ is simply the inverse,
$$
V^{-1}=\left[
\begin{array}{ccc}
-2&1&2\\
3&-1&-2\\
2&-1&-1
\end{array}
\right]_{\Epsilon_3,J}.
$$
Hence the matrix representation of $L$ w.r.t. $J$
$$
\begin{aligned}
B = \text{Rep}_{J}(L)&=V^{-1}AV\\&=
\left[
\begin{array}{ccc}
0&0&0\\
0&1&0\\
0&0&1
\end{array}
\right]_{J}.
\end{aligned}
$$

## 9

Taking transpose on both sides of the first equation,
$$
x^TA^T=0.
$$
Multiply both sides by $y,$
$$
x^TA^Ty=0. \label1
$$
Now combine $(\ref1)$ with the second equation given,
$$
x^T2y=2x^Ty=0\implies x^Ty=0\implies x\perp y.​
$$

## 10

### (a)

True.
$$
Q \text{ orthogonal}\implies QQ^T=I\implies (Q^T)^{-1}Q^{-1}=(Q^{-1})^TQ^{-1}=I\implies Q^{-1} \text{ orthogonal.}
$$

#### Example.

$Q=I$ is orthogonal, $Q^{-1}=I$ is also orthogonal.

### (b)

True.
$$
\begin{aligned}
||Qx||^2&=(Qx)^TQx\\
&=x^TQ^TQx\\&=
x^T
\left[
\begin{array}{c}
q_1^T\\
q_2^T\\
\vdots\\
q_n^T
\end{array}
\right]
[q_1,q_2,...,q_n]x\\&=
x^T\left[
\begin{array}{c}
q_1^Tq_1& q_1^Tq_2&...&q_1^Tq_n\\
q_2^Tq_1&q_2^Tq_2&...&q_2^Tq_n\\
\vdots&\vdots&\ddots&\vdots\\
q_n^Tq_1&q_n^Tq_2&...&q_n^Tq_n
\end{array}
\right]x\\&=
x^TI_nx\\&=||x||^2,
\end{aligned}
$$
and hence the square roots remain equal.

#### Example.

$Q=\left[
\begin{array}{cc}
1 &0\\
0& 1\\
0& 0
\end{array}
\right],Qx=\left[
\begin{array}{cc}
1 &0\\
0& 1\\
0& 0
\end{array}
\right]\left[
\begin{array}{c}
x_1\\
x_2\end{array}
\right]=\left[
\begin{array}{c}
x_1\\
x_2\\
0
\end{array}
\right].$ $||Qx||=||x||=\sqrt{x_1^2+x_2^2}$.

### (c)

False.

#### Example.

For the same $Q$ as in (b), $Q^Ty=\left[
\begin{array}{cc}
1 &0\\
0& 1\\
0& 0
\end{array}
\right]^T\left[
\begin{array}{c}
y_1\\
y_2\\y_3 \end{array}
\right]=\left[
\begin{array}{c}
y_1\\
y_2\\
\end{array}
\right].$ Clearly $||Q^Ty||=\sqrt{y_1^2+y_2^2}\neq\sqrt{y_1^2+y_2^2+y_3^2}=||y||.$

## 11

$\forall p\in P(\R) \forall q \in W_2,$ their inner product
$$
\begin{aligned}
\langle p,q\rangle&=\int_{-1}^1p(x)q(x)dx\\&=
\int_{-1}^1(a_0+a_1x+a_2x^2+a_3x^3+...)(b_0+b_2x^2+...)dx\\&=
\int_{-1}^1\sum _{n=0}^\infty c_nx^{2n+1}dx+\int_{-1}^1\sum _{n=0}^\infty d_n x^{2n}dx\\&=
\sum _{n=0}^\infty\int_{-1}^1c_nx^{2n+1}dx+\sum _{n=0}^\infty\int_{-1}^1d_nx^{2n}dx\\&=
\sum _{n=0}^\infty\int_{-1}^1d_nx^{2n}dx
\end{aligned}
$$
where
$$
d_n=\sum_{i+2j=2n}a_ib_{2j}.
$$
For the inner product to be $0,$ it is equivalent to ask $d_n=0,$ i.e. 
$$
a_0b_0=a_0b_2+a_2b_0=a_0b_4+a_2b_2+a_4b_0=...=0
$$
for any choice of $b_0,b_2,b_4...$ Clearly this is true if and only if
$$
a_0=a_2=a_4=...=0.
$$
That is, $p \in W_1.$ Thus we conclude in $P( \R),$
$$
\forall q \in W_2,p\perp q \iff p\in W_1.
$$
In other words,
$$
W_1=W_2^{\perp}.
$$

## 12

To find the first basis vector $s:=[ s_1,s_2,s_3]^T,$ we use the fact that
$$
\langle s,[1,2,-5]^T\rangle=0.
$$
Equivalently,
$$
s_1+2s_2-5s_3=0.\label2
$$
A particular solution to $(\ref2)$ is
$$
s=[s_1,s_2,s_3]^T=[1,2,1]^T.
$$
To find the other basis vector $u,$ we simply use the cross product
$$
u = s\times [1,2,-5]^T=[-12,6,0].
$$
Normalizing,
$$
\bar s= \frac{1}{\sqrt6}[1,2,1]^T,\\
\bar u= \frac{1}{\sqrt5}[-2,1,0]^T.
$$
An orthonormal basis for $U$:
$$
B_U = \{\bar s, \bar u\}.
$$

## 13

We want:
$$
\begin{aligned}
4&=C+D(-2)\\
2&=C+D(-1)\\
-1&=C+D(0)\\
0&=C+D(1)\\
0&=C+D(2)
\end{aligned}
$$
In matrix form,
$$
b=
\left[
\begin{array}{cc}
4\\
2\\
-1\\
0\\
0
\end{array}
\right]=\underbrace{\left[
\begin{array}{c}
1&-2\\
1&-1\\
1&0\\
1&1\\
1&2
\end{array}
\right]}_{A}\underbrace{
\left[
\begin{array}{c}
C\\
D\end{array}
\right]}_x.
$$
The normal equation is $A^Tb=A^TAx:$
$$
\left[
\begin{array}{c}
5\\
-10\end{array}
\right]=\left[
\begin{array}{cc}
5&0\\
0&10\end{array}
\right]x.\label5
$$
Solving $(\ref5),$
$$
x=[1,-1]^T.
$$
Hence the best fitting line in the sense of least-square is:
$$
{y= 1-t}.
$$

## 14

```julia
using Plots
n = BigInt(100); p = .1; λ=10; x = BigInt.(0:n);
bi(x) = binomial(n, x) * p^x * (1 - p)^(n - x)
poi(x) = λ^x / (ℯ^λ * factorial(x))
x1 = plot(bi.(x), line=:stem, label="Binomial(100, 0.1)")
x2 = plot(poi.(x), line=:stem, label="Poisson(10)", color=:red)
plot(x1,x2)
```

![pmf](C:\Users\chen1\Documents\CSC\MAT2040\A04\pmf.png)

## 15

```julia
using Plots
n = 10; N = 1e6; p = .5
rslt = Dict([(x,0) for x=0:n])
for _ = 1:N
    c = count(x->(rand()<p), 1:n)
    rslt[c] += 1
end
f(x) = rslt[x] / N
bi(x) = binomial(n, x) * p^x * (1 - p)^(n - x)
emp = plot(f, 0:n, line=:stem, marker=:auto, label="Empirical")
theo = plot(bi, 0:n, line=:stem, marker=:circle, label="Theoretical", color=:red)
plot(emp, theo)
```

![pmf2](C:\Users\chen1\Documents\CSC\MAT2040\A04\pmf2.png)

## 16

### (a)

```julia
julia> using LinearAlgebra

julia> U=rand(4,4); V=rand(4,4); b=ones(4,1);

julia> V
4×4 Array{Float64,2}:
 0.583791  0.504311  0.33405   0.544164
 0.211525  0.545206  0.658239  0.913852
 0.514587  0.826214  0.768501  0.783236
 0.203812  0.794052  0.68816   0.697757

julia> rank(V)
4
```

### (b)

```julia
julia> ϵ₄E = inv(U) # change-of-basis matrix from ϵ₄ to E
4×4 Array{Float64,2}:
  4.57283   5.16672  -1.57722  -4.54252
 -3.21203  -5.55121   2.9037    2.59196
  3.35517  11.4939   -2.21768  -7.08184
 -4.51906  -8.71237   1.52479   8.35272

julia> ϵ₄F = inv(V) # change-of-basis matirx from ϵ₄ to F
4×4 Array{Float64,2}:
  0.50853  -0.467216   2.65237  -2.76198
  2.55582  -2.06237   -4.19896   5.42121
 -5.31879   0.190387   7.94312  -5.01755
  2.18856   2.29569   -3.83018   1.01909
```

## 17

### (a)

```julia
julia> using LinearAlgebra

julia> c = ϵ₄E * b # representation of b w.r.t. E
4×1 Array{Float64,2}:
  3.61980629652735
 -3.2675806363290505
  5.549578103410754
 -3.353928644644931

julia> d = ϵ₄F * b # representation of b w.r.t. F
4×1 Array{Float64,2}:
 -0.06829308417488722
  1.715707100063645
 -2.202830488545467
  1.6731589608599862

julia> norm(b - U*c)
1.041481514324134e-15

julia> norm(b - V*d)
4.965068306494546e-16
```

### (b)

```julia
julia> S = ϵ₄F*U # change-of-basis matrix from E to F
4×4 Array{Float64,2}:
  1.08202    1.73673   0.276038  -0.0471195
  0.481655  -2.27235  -0.751135   0.979271
  0.446511   4.47906   1.831     -0.195379
 -0.555176  -2.02403  -0.527645   0.000801451

julia> T = ϵ₄E*V # change-of-basis matrix from F to E
4×4 Array{Float64,2}:
  2.02503   0.212938   0.590411   2.80508
 -1.0269   -0.189189  -0.711821  -2.73802
  1.80543   0.502986   2.10881    5.65114
 -1.99406   0.863238  -0.324607  -3.3985

julia> norm(d - S*c)
6.139584144267543e-15

julia> norm(c - T*d)
7.768388458966724e-15
```

## 18

```julia
julia> using LinearAlgebra

julia> A = [3 5;-2 -6;-5 -11]; b = [3; 3; 3];

julia> # A'Ax* = A'b ⟺ x* = inv(A'A)A'b

julia> x_ast = inv(A'*A)*A'*b
2-element Array{Float64,1}:
  3.9999999999999982
 -1.9999999999999991

julia> norm(A*x_ast - b)
1.7320508075688794
```



