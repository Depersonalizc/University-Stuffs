# Assignment 5

## Table of contents

[TOC]

## 1

Characteristic polynomial of $A$ is
$$
p_A(\lambda)=\lambda^2+\lambda-6=(\lambda+3)(\lambda-2).
$$
Hence the eigenvalues for $A$ are $\lambda_1=-3, \lambda_2=2.$

Let
$$
(\lambda I-A)x=0,
$$
yielding eigenvectors for $A:$
$$
x_1=[-3,2]^T,x_2=[1,1]^T.
$$
Characteristic polynomial of $A^2$ is
$$
p_{A^2}(\lambda)=\lambda^2-13\lambda+36=(\lambda-9)(\lambda-4).
$$
Hence the eigenvalues for $B$ are $\lambda'_1=9, \lambda'_2=4.$

Let
$$
(\lambda' I-A^2)x=0,
$$
yielding eigenvectors for $A^2:$
$$
x_1=[-3,2]^T,x_2=[1,1]^T.
$$
$A^2$ has the same ==***eigenvectors***== as $A.$ When $A$ has eigenvalues $\lambda_1$ and $\lambda_2,$ $A^2$ has eigenvalues ==$\lambda_1^2,\lambda_2^2.$== In this example, ==$\lambda_1^2+\lambda_2^2=13=\text{tr}(A^2).$==  

## 2

Characteristic polynomial of $A :$
$$
\begin{aligned}

p_A(\lambda)&=\left|
\begin{array}{cc}
\lambda I-B&-C\\
0&\lambda I -D
\end{array}
\right|\\&=
\left|
\left[\begin{array}{}
I& 0\\
0&\lambda I - D
\end{array}\right]
\left[\begin{array}{}
\lambda I - B& -C\\
0&I
\end{array}\right]
\right|\\&=
\left|
\begin{array}{}
I& 0\\
0&\lambda I - D
\end{array}
\right|\cdot\left|
\begin{array}{}
I&-C\\
0&\lambda I -B
\end{array}
\right|\\&=
|\lambda I-D|\cdot|\lambda I-B|\\&=
(p_D\cdot p_B)(\lambda).
\end{aligned}
$$
Hence the eigenvalues of $A$ are exactly those of $B$ and $D,$ namely $\lambda_A=1,2,5,7.$

## 3

==$1\neq n=2.$==

## 4

$$
p_A(\lambda)=\lambda^2-25\lambda=\lambda(\lambda-25).
$$

The eigenvalues of $A$ are $\lambda_1=0,\lambda_2=25.$

Let $(\lambda I-A)x=0.$ A specific solution is $x_1=[-4/5,3/5], x_2=[3/5,4/5]^T.$

Hence there are exactly $8$ orthogonal matrices that diagonalize $A: $
$$
Q_{1,2}=\left[\begin{array}{cc}
\pm4/5&3/5\\
\mp3/5&4/5
\end{array}\right],Q_{3,4}=\left[\begin{array}{cc}
\pm4/5&-3/5\\
\mp3/5&-4/5
\end{array}\right],\\Q_{5,6}=\left[\begin{array}{cc}
3/5&\pm4/5\\
4/5&\mp3/5
\end{array}\right],Q_{7,8}=\left[\begin{array}{cc}
-3/5&\pm4/5\\
-4/5&\mp3/5
\end{array}\right].
$$

## 5

If $\lambda = a+ib$ is an eigenvalue of a real matrix $A,$ then $Ax=\lambda x$ for some $x\neq0.$  Then
$$
A\overline x=\overline{(Ax)}=\overline{(\lambda x)}=\overline\lambda \overline x.
$$
Thus $\overline\lambda$ is also an eigenvalue of $A,$ corresponding to the eigenvector $\overline x.$

From the previous proven proposition, any real $3\times3$ matrix $A$ has characteristic polynomial of the form
$$
p_{A}(\lambda)=(\lambda-\lambda_1)(\lambda-\overline\lambda_1)(\lambda-\lambda_2).
$$
Note that the constant term in the polynomial is $r\cdot\lambda_2,$ where $r=- \lambda_1\overline\lambda_1\in\R.$ But $p_A(\lambda)= |\lambda I-A_{\R^{3\times3}}|,$ hence the constant term must be real, forcing $\lambda_2$ also to be real.

## 6

$$
B=S^{-1}AS\implies SA=BS\implies S(A-\lambda I)=(B-\lambda I)S.
$$

Since $|S|\neq0,$ it follows that $A$ and $B$ has the same characteristic polynomial:
$$
p_A(\lambda)=p_B(\lambda).
$$
Thus $A$ and $B$ has the same eigenvalues and hence same diagonalization.
$$
Q_A^T A Q_A=\Lambda=Q_B^TBQ_B\implies B=(Q_BQ_A^T)A(Q_AQ_B^T).\label5
$$
The proof is done by noting $M:=Q_BQ_A^T$ as an orthonormal matrix, and from $(\ref5)$ we have
$$
B=MAM^T.
$$

## 7

The characteristic polynomial of $A :$
$$
p_A(\lambda)=(\lambda-\cos\theta)^2+\sin^2\theta=\lambda^2-2\cos\theta\cdot\lambda+1.
$$
has complex roots $e^{\pm i\theta},$ also being eigenvalues of $A$ corresponding to the eigenvectors $[1,\pm i]^T,$ whenever $\theta\neq k\pi, k\in\Z.$

The geometric interpretation of the result is that $A$ corresponds to an anticlockwise rotation (denoted as $R$) in $\C^2$ by an angle of $\theta,$ i.e. in the basis $\{[1,\pm i]^T\},$
$$
R([x,y]^T)=[e^{i\theta}x,e^{-i\theta}y]^T.
$$

## 8

### (a)

$$
x^Tx=x^TQ^TQx=(Qx)^TQx=(\lambda x^T)(\lambda x)=\lambda^2(x^Tx)\implies\lambda^2=1.
$$

Equivalently,
$$
|\lambda|=1.
$$

### (b)

$$
QQ^T=I\implies |QQ^T|=|Q||Q^T|=|Q|^2=1.
$$

Equivalently
$$
|\det(Q)|=1.
$$

## 9

$$
A(Sx)=ASx=(SBS^{-1}S)x=SBx=S(\lambda x)=\lambda(Sx).
$$

## 10

### (a)

$$
<z_1,z_2>=z_2^Hz_1=\frac{1-i}{2\sqrt2}+\frac{-(1-i)}{2\sqrt2}=0.
$$

$$
{<z_1,z_1>}={(2+2)/4}=1
$$

$$
{<z_2,z_2>}={(1+1)/2}=1.
$$

### (b)

$$
z=4z_1+2\sqrt2z_2.
$$

## 11

### (a)

$$
u_1^Hz=(4+2i)u_1^Hu_1=4+2i\\z^Hu_1=(u_1^Hz)^H=4-2i.
$$

$$
u_2^Hz=(6-5i)u_2^Hu_2=6-5i\\z^Hu_2=(u_2^Hz)^H=6+5i.
$$

### (b)

$$
\begin{aligned}
||z||&=\sqrt{z^Hz}\\&=\sqrt{[4-2i,6+5i]\left[\begin{array}{}4+2i\\6-5i\end{array}\right]}\\&=\sqrt{20+61}\\&=9.

\end{aligned}
$$

## 12

(a); (c).

## 13

$$
U^H=\overline{(I-2uu^H)^T}=\overline{I-2\overline{u}u^T}=I-2uu^H=U.
$$

Therefore $U$ is Hermitian. Further,
$$
UU^H=U^HU=(I-2uu^H)^2=I-4uu^H+4(uu^H)^2=I.
$$
Hence $U$ is also unitary and, consequently,
$$
U^{-1}=U^H=U.
$$

## 14

```julia
julia> using LinearAlgebra

julia> A = [1 2;3 4];
```

### (a)

Suppose $A_{m\times n}.$ Then
$$
||A||^2_F=\sum_{j=1}^n\sum_{i=1}^ma^2_{ij}=\underbrace{\sum_{j=1}^n(a_j)^Ta_j}_{a_j \text{ is the }j^\text{th} \\\text{column of }A}=\text{tr}(A^TA).
$$
Taking the square root on both sides yields
$$
||A||_F=\sqrt{\text{tr}(A^TA)}
$$
as desired.



```julia
julia> tr(A'A) == tr(A*A') == sum((x -> x^2).(A))
true
```

### (b)

```julia
julia> eigen(A'A).values
2-element Array{Float64,1}:
  0.13393125268149486
 29.866068747318508
```

## 15

```julia
julia> using LinearAlgebra
```

### (a)

$$
x_{k+1}=\left[\begin{array}{c}
g_{k+2}\\
g_{k+1}
\end{array}\right]=
\underbrace{\left[\begin{array}{cc}
1-w&w\\
1&0
\end{array}\right]}_A
\left[\begin{array}{c}
g_{k+1}\\
g_k
\end{array}\right]=Ax_k.
$$

### (b)

$$
p_A(\lambda)=\lambda^2+(w-1)\lambda-w=(\lambda+w)(\lambda-1).
$$

Hence the eigenvalues of $A$ are $\lambda_1=-w, \lambda_2=1.$

Let $(A-\lambda I)x=0,$ we have corresponding eigenvectors $x_1=1/\sqrt{w^2+1}\cdot[-w,1]^T,x_2=1/\sqrt2\cdot[1,1]^T.$

```julia
julia> w = .1;

julia> A = [1 - w w
            1 0];

julia> v1, v2 = [1;1] / sqrt(2), [-w;1] / sqrt(w^2 + 1);

julia> vals = [1; -w]; vecs = [v1 v2];

julia> (eigen(A).values, eigen(A).vectors) == (vals, vecs)
true
```

### (c)

$\lambda_{1,2}\to\mp 1$ respectively; $x_{1,2} \to 1/\sqrt2\cdot[\pm1, 1]^T$ respectively. 

$\{x_1,x_2\}$ forms an orthonormal basis in the limit.

For $w=-1, x_1=x_2=1/\sqrt2\cdot[1,1]^T.$ $\{x_1,x_2\}$ is linearly dependent and therefore does not form a basis. For the same reason, $[x_1,x_2]$ is non-invertible. Hence by definition $A$ is no longer diagonalizable.

### (d)

$$
x_k=A^kx_0=S\Lambda^kS^{-1}x_0.\label1
$$

The columns of S are the eigenvectors of $A.$ Now as $k\to \infty, \Lambda^k=\text{diag}[(-w)^k,1]\to \text{diag}(0,1)$ given $0<w<1.$ Hence $g_k$ always converges to some constant (possibly zero.) In fact, as to be shown in $(\ref2),g_k\to(1+w)^{-1}(wg_0+g_1).$



### (e)

$$
A^k=S\Lambda^kS^{-1}\to\frac1{1+w}\left[\begin{array}{cc}
1 & w\\
1 & w
\end{array}\right]=:B.\label3
$$

### (f)

Using $(\ref1)$ and $(\ref3),$
$$
g_k\to (Bx_0)_{11}=\frac1{1+w}(wg_0+g_1).\label2
$$
Plugging in the initial condition,
$$
g_k \to\frac1{1+0.5}(0.5\cdot 0+1)=\frac23.
$$

### (g)

By $(\ref1)$ and the initial condition,
$$
g_k=\frac23[(-1)^{k+1}2^{-k}+1],
$$
and thus
$$
\left|\frac{g_{k+1}-2/3}{g_k-2/3}\right|=\frac{2^{-k-1}}{2^{-k}}=\frac12.
$$
In other words,
$$
|g_k-2/3|\sim (1/2)^{k}.
$$
This is verified by the numerical computation: 

```julia
julia> n = 0:24;

julia> error(x) = abs(2/3*((-1)^(x+1)*.5^x)); # |gₖ - 2/3|

julia> expo(x) = .5^x;

julia> error.(n) ./ expo.(n) # |gₖ - 2/3| ∝ (1/2) ^ k
25-element Array{Float64,1}:
 0.6666666666666666
 0.6666666666666666
 0.6666666666666666
 0.6666666666666666
 0.6666666666666666
 ⋮
 0.6666666666666666
 0.6666666666666666
 0.6666666666666666
 0.6666666666666666
 0.6666666666666666
```

## 16

The (post-multiplying) transition matrix:
$$
\bold P = 
\left[\begin{array}{}

.9 & .1 &0\\
0 & 0 & 1\\
1 & 0 & 0


\end{array}\right].
$$
The steady-state probabilities are given numerically by

```julia
julia> P=[.9 .1 0; 0 0 1; 1 0 0];

julia> P^100
3×3 Array{Float64,2}:
 0.833333  0.0833333  0.0833333
 0.833333  0.0833333  0.0833333
 0.833333  0.0833333  0.0833333
```

That is, with any initial probabilities $\bold x^{(0)}=\left[\begin{array}{}p_0^{(0)}&p_1^{(0)}&p_2^{(0)}\end{array}\right],$
$$
\lim_{n\to\infty}\bold x^{(n)}=\bold x^{(0)}\lim_{n\to\infty} \bold P^n=\left[\begin{array}{}

5/6 & .5/6 &.5/6\\


\end{array}\right].
$$

## 17

The (post-multiplying) transition matrix:
$$
\bold P =\left[\begin{array}{}

.5 & .5 &0&0\\
.04 & .9 & .06&0\\
.04 & 0 & .9&.06\\
.04 & .02 & .04&.9

\end{array}\right].
$$

```julia
julia> P=[.5 .5 0 0
       .04 .9 .06 0
       .04 0 .9 .06
       .04 .02 .04 .9];

julia> P^1000
4×4 Array{Float64,2}:
 0.0740741  0.40913  0.322997  0.193798
 0.0740741  0.40913  0.322997  0.193798
 0.0740741  0.40913  0.322997  0.193798
 0.0740741  0.40913  0.322997  0.193798
```

Hence given any initial probabilities $\bold x^{(0)}=\left[\begin{array}{}p_0^{(0)}&p_1^{(0)}&p_2^{(0)}\end{array}\right],$
$$
\lim_{n\to\infty}\bold x^{(n)}=\bold x^{(0)}\lim_{n\to\infty} \bold P^n\approx\left[\begin{array}{}

.074 & .409 &.323 &.194\\


\end{array}\right].
$$
