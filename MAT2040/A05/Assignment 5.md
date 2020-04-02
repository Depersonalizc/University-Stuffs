# Assignment 5

## Table of contents

[TOC]

## 1

### (a)

By the fact that
$$
\text{rank }AA^T=\text{rank }A^TA
$$
and the rank-nullity theorem, one has
$$
\text{rank }A-\text{rank }{AA^T}=\text{nullity }A^TA-\text{nullity }A.
$$
Now from the proven fact that 
$$
\text{Null }A=\text{Null } A^TA\implies \text{nullity }A=\text{nullity }A^TA,
$$
one obtains
$$
\text{rank }A=\text{rank }{AA^T}.\label1
$$

### (b)

$$
AA^T=[a_1,a_2,...,a_n]A^T
$$

where $a_i$ is the $i$-th column of $A$. Note that every column of $AA^T$ is a linear combination of $a_i$'s. Hence
$$
\text{Col }AA^T\sub \text{Col }A.
$$
But from $(\ref1)$ we know that the dimensions of two column space are equal, so it must be the case that
$$
\text{Col }AA^T= \text{Col }A.
$$

## 2

### (a)

The normal equation $A^TAx=A^Tb$ is:
$$
\left[
\begin{array}{cc}
7&1\\
1&6\end{array}
\right]x=\left[
\begin{array}{c}
8\\
7\end{array}
\right],
$$
which has a unique solution since $A^TA$ is invertible.

### (b)

$$
x=\left[
\begin{array}{cc}
7&1\\
1&6\end{array}
\right]^{-1}\left[
\begin{array}{c}
8\\
7\end{array}
\right]=\left[
\begin{array}{c}
1\\
1\end{array}
\right].
$$

### (c)

$$
\hat b=A(A^TA)^{-1}A^Tb=[3,1,2,-1]^T.
$$

### (d)

Let
$$
A^Tx=0.
$$
Solving the system,
$$
x=s[3,1,-5,0]^T+t[1,2,0,5]^T\ \ \ \ s,t\in\R.
$$
Define
$$
B:=\left[
\begin{array}{c}
3&1\\
1&2\\
-5&0\\
0&5
\end{array}
\right].
$$
Then $\text{Col } B=\text{Null }A^T,$ and
$$
\tilde b=B(B^TB)^{-1}B^Tb=[1,0,-2,-1]^T.
$$

## 3

### (a)

$$
[p_1\  p_2\ p_3\ p_4]
\left[
\begin{array}{c}
p_1\\
p_2\\
p_3\\
p_4
\end{array}
\right]=\left[
\begin{array}{c}
8&2&4\\
2&2&2\\
4&2&4
\end{array}
\right]

\text{~}
\left[
\begin{array}{c}
1&0&0\\
0&1&0\\
0&0&1
\end{array}
\right].
$$

Hence $\det (\cdot)=\det(I_3)=1\neq0,$ implying $(\cdot)$ is invertible.

### (b)

$$
G=AP\implies GP^T=APP^T.\label2
$$

From (a) we know $PP^T$ is invertible, thus $(\ref2)$ implies
$$
A=GP^T(PP^T)^{-1}=\left[
\begin{array}{c}
0&-1&0\\
1&0&0\\
0&0&1
\end{array}
\right].
$$

## 4

$$
\begin{aligned}
<\sin(x),\cos(x)>&=\int_0^{\pi}\sin(x)\cos(x)dx\\&=
\frac12\int_0^{\pi}\sin(2x)dx\\&=
-\cos(2x)|_0^\pi\\&=0.
\end{aligned}
$$

## 5

### (a)

$$
\{[1,0]^T,[0,1]^T\}
$$

### (b)

$$
\{[1,0,0,0]^T,[0,1,0,0]^T,[0,0,1,0]^T,[0,0,0,1]^T\}
$$

## 6

### (a)

$$
\begin{aligned}
\frac2\pi\int_0^{\frac\pi2}r(t)\sqrt2\cos(1000t)\ dt&=\frac4\pi\int_0^{\frac\pi2}A\cos^2(1000t)+B\sin(1000t)\cos(1000t)\ dt\\&=
\frac4\pi\int_0^{\frac\pi2}A\cos^2(1000t)\ dt\\&=\frac{2A}\pi\int_0^{\frac\pi2}1+\cos(2000t)\ dt\\&=
\frac{2A}\pi \cdot\frac\pi2\\&=A.
\end{aligned}
$$

### (b)

$$
\begin{aligned}
\frac2\pi\int_0^{\frac\pi2}r(t)\sqrt2\sin(1000t)\ dt&=\frac4\pi\int_0^{\frac\pi2}B\sin^2(1000t)+A\sin(1000t)\cos(1000t)\ dt\\&=
\frac4\pi\int_0^{\frac\pi2}B\sin^2(1000t)\ dt\\&=\frac{2B}\pi\int_0^{\frac\pi2}1-\cos(2000t)\ dt\\&=
\frac{2B}\pi \cdot\frac\pi2\\&=B.
\end{aligned}
$$

## 7

Fourier coefficients:
$$
a_n=\frac2P\int_Pf(x)\cdot\cos(2\pi x\frac nP)\ dx
$$

$$
b_n=\frac2P\int_Pf(x)\cdot\sin(2\pi x \frac nP)\ dx.
$$

### (a)

$P=2\pi.$
$$
\begin{aligned}
a_n&=\frac1\pi\int_{-\pi}^{\pi}\cos(x)\cdot\cos(nx)\ dx\\&=\begin{cases}
\left.x/2+\sin(2x)\right\vert_{-\pi}^{\pi}=1,   \ \ n=1\\
\left.\frac{2n{\sin}(\pi n)}{\pi (n^2-1)}\right\vert_{\pi}^{-\pi}=0, \ \ n\neq1.
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
b_n&=\frac1\pi\int_{-\pi}^{\pi}\cos(x)\cdot\sin(nx)\ dx\\&=
0.
\end{aligned}
$$

Hence the Fourier Series for $f(x)=\cos (x)$ is simply $f(x)$.

### (b)

$P=2\pi$.
$$
\begin{aligned}
a_n&=\frac1\pi\int_{-\pi}^{\pi}[2\cos(x)+4\cos(2x)]\cdot\cos(nx)\ dx\\&=\begin{cases}
2,   \ \ n=1\\
4,\ \ n=2\\
0,\ \ \text{otherwise.}
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
b_n&=\frac1\pi\int_{-\pi}^{\pi}[2\cos(x)+4\cos(2x)]\cdot\sin(nx)\ dx\\&=
0.
\end{aligned}
$$

The Fourier Series for $f(x)=2\cos(x)+4\cos(2x)$ is given by $f(x).$

### (c)

$P=2\pi.$
$$
\begin{aligned}
a_n&=\frac1\pi\int_{-\pi}^{\pi}f(x)\cdot\cos(nx)\ dx\\&=
\frac A\pi\int_{0}^{\pi}\cos(nx)\ dx\\&=\begin{cases}
A, \ \ n=0\\
\left.\frac{A}{\pi n}\sin(nx)\right\vert_0^\pi=0, \ \ n\neq0.

\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
b_n&=\frac1\pi\int_{-\pi}^{\pi}f(x)\cdot\sin(nx)\ dx\\&=
\frac A\pi\int_{0}^{\pi}\sin(nx)\ dx\\&=\begin{cases}
0, \ \ n=0\\
\frac{A}{\pi n}[1-\cos(\pi n)], \ \ n\neq0.
\end{cases}\\&=\begin{cases}
0, \ \ n \text{ is even}\\
\frac{2A}{\pi n}, \ \ n\text{ is odd}.
\end{cases}
\end{aligned}
$$

The Fourier Series for $f(x)$ is given by
$$
\frac A2+\sum_{n=1}^\infty b_n\sin(nx)
$$
where
$$
b_n=\begin{cases}
0, \ \ n \text{ is even}\\
\frac{2A}{\pi n}, \ \ n\text{ is odd}.
\end{cases}
$$

## 8

### (a)

$$
\begin{aligned}
F[n]&=\sum_{k=0}^{5}f[k]e^{-\frac{i\pi}{3}kn},n=0:5\\&=
[0,0,0,6,0,0].
\end{aligned}
$$

The transformation matrix
$$
\begin{aligned}
\bold F&=[e^{-\frac {i\pi}{3}j\cdot k}],\ j,k=0:5\\&=
\left[
\begin{array}{cccc}
1&1&1&1&1&1\\
1&e^{-i\pi/3}&e^{-2i\pi/3}&e^{-3i\pi/3}&e^{-4i\pi/3}&e^{-5i\pi/3}\\
1&e^{-2i\pi/3}&e^{-4i\pi/3}&e^{-6i\pi/3}&e^{-8i\pi/3}&e^{-10i\pi/3}\\
1&e^{-3i\pi/3}&e^{-6i\pi/3}&e^{-9i\pi/3}&e^{-12i\pi/3}&e^{-15i\pi/3}\\
1&e^{-4i\pi/3}&e^{-8i\pi/3}&e^{-12i\pi/3}&e^{-16i\pi/3}&e^{-20i\pi/3}\\
1&e^{-5i\pi/3}&e^{-10i\pi/3}&e^{-15i\pi/3}&e^{-20i\pi/3}&e^{-25i\pi/3}
\end{array}
\right]
\end{aligned}.
$$

### (b)

The inverse transformation matrix
$$
\bold F^{-1}=\frac16\bold F^*=[e^{\frac {i\pi}3j\cdot k}],\ j,k=0:5.
$$
Conducting on $F[k],$
$$
\bold {F^{-1}}F[k]^T=[1,-1,1,-1,1,-1]^T.
$$

## 9

### (a)

$$
a_1=[2,1,2]^T
$$

$$
a_2=[1,1,1]^T.
$$

Normalizing $a_1,$
$$
n_1:=\frac{a_1}{||a_1||}=\frac1{3}[2,1,2]^T.
$$
Then let
$$
\begin{aligned}
a_2'&=a_2-<a_2,n_1>n_1\\&=
[1,1,1]^T-5/9\cdot[2,1,2]^T\\&=
[-1/9,4/9,-1/9]^T.
\end{aligned}
$$
Normalizing $a_2',$
$$
n_2:=\frac{a_2}{||a_2||}=\frac1{\sqrt{18}}[-1,4,-1]^T.
$$
Then $B:=\{n_1, n_2\}$ is an orthonormal basis for $\text{Col }A.$

### (b)

$$
A=\underbrace{\left[
\begin{array}{cc}
2/3&-1/\sqrt18\\
1/3&4/\sqrt18\\
2/3&-1/\sqrt18
\end{array}
\right]}_Q
\underbrace{\left[
\begin{array}{cc}
3&5/3\\
0&\sqrt2/3
\end{array}
\right]}_R.\label3
$$

### (c)

Rewriting normal equation $A^TAx=A^Tb$ using $(\ref3)$, 
$$
R^T(Q^TQ)Rx=R^TRx=R^TQ^Tb.
$$
Hence $Rx=Q^Tb,$
$$
x=R^{-1}Q^Tb=[9,-3]^T.
$$

## 10

### (a)

$$
h_1=[0.3, 0.4]^T
$$

$$
h_2=[0.8,0.6]^T
$$

First normalizing $h_1,$
$$
n_1:=\frac{h_1}{||h_1||}=\frac15[3,4]^T.
$$
Let
$$
\begin{aligned}
h_2'&:=h_2-<h_2,n_1>n_1\\&=
[0.8,0.6]^T-\frac{4.8}{25}[3,4]^T\\&=
[0.224,-0.168].
\end{aligned}
$$
Normalizing $h_2',$
$$
n_2:=\frac{h_2'}{||h_2'||}=\frac15[4,-3]^T.
$$
Hence $B:=\{n_1,n_2\}$ is an orthonormal basis for $\text{Col }H.$

### (b)

$$
H=\underbrace{\left[
\begin{array}{cc}
3/5&4/5\\
4/5&-3/5
\end{array}
\right]}_Q
\underbrace{\left[
\begin{array}{cc}
1/2&24/25\\
0&7/25
\end{array}
\right]}_R.\label4
$$

### (c)

Rewriting the equation $y=Hx+n$ using $(\ref4),$ 
$$
[x_1,x_2]^T=R^{-1}Q^T(y-n)=[1,-1]^T.
$$

## 11

### (a)

$\det(AB)=\det(A)\cdot \det(B)=12.$

### (b)

$\det(5A)=5^3\det(A)=-375.$

### (c)

$\det(B^T)=\det(B)=-4.$

### (d)

$\det(A^{-1})=\det^{-1}(A)=-\frac13.$

### (e)

$\det(A^3)=\det^3(A)=-27.$

## 12

### (a)

$\det(A)=3C_{11}+1C_{12}+(-2)C_{13}=0+6+0=6.$

### (b)

$\det(A^4)=\det^4(A)=6^4=1296.$

### (c)

$$
\begin{aligned}
\text{adj}(A)&=[c_{ij}]^T\\&=\left[
\begin{array}{ccc}
0&6&0\\
-6&28&5\\
0&2&1
\end{array}
\right]^T\\&=\left[
\begin{array}{ccc}
0&-6&0\\
6&28&2\\
0&5&1
\end{array}
\right].

\end{aligned}
$$

### (d)

$$
x=A^{-1}[16,-2,-8]^T=[2,4,-3]^T.
$$

### (e)

By Cramer's Rule,
$$
x=\frac{\text{adj}(A)}{\det(A)}[16,-2,-8]^T=[2,4,-3]^T.
$$

## 13

```julia
julia> using LinearAlgebra
julia> A, B = rand(5,5), rand(5,5);
```

### (i)

```julia
julia> det(A*B)-det(A)det(B)
1.8431436932253575e-18
```

### (ii)

```julia
julia> det(A)-det(A')
-3.469446951953614e-18
```

### (iii)

```julia
julia> C = A;

julia> for i = 1:5
           C[2,i],C[4,i]=C[4,i],C[2,i]
           end

julia> det(C)-det(A)
0.0
```

## 14

```julia
julia> using LinearAlgebra

julia> A = rand(4,4); L,U,p = lu(A);P = zeros(4,4);

julia> for i = 1:4
           P[i,p[i]] = 1
           end
```

### (i)

```julia
julia> det(A)
-0.16431584484935427

julia> det(P)
-1.0

julia> det(U)
0.16431584484935427

julia> det(L)
1.0
```

### (ii)

$$
\det (P)\cdot\det (A)=\det (U).
$$

### (iii)

```julia
julia> A[1,1]=A[2,1]=0; L,U,p = lu(A);P = zeros(4,4);

julia> for i = 1:4
           P[i,p[i]]=1
           end

julia> det(A)
-0.19272624085598167

julia> det(P)
1.0

julia> det(U)
-0.19272624085598167

julia> det(L)
1.0
```

## 15

```julia
julia> using LinearAlgebra

julia> A = rand(4,4); b = rand(4,1);

julia> B(j) = hcat([i==j ? b : A[:,i] for i = 1:4]...)
B (generic function with 1 method)

julia> x(j) = det(B(j)) / det(A)
x (generic function with 1 method)

julia> x_julia = A\b; x_cramer = [x(i) for i = 1:4];

julia> norm(x_julia-x_cramer)
3.3766115072321297e-16
```

## 16

```julia
julia> using LinearAlgebra

julia> n = 4; A = rand(n,n); val,vect = eigen(A);
```

### (i)

```julia
julia> prod(val)-det(A)
3.122502256758253e-17
```

### (ii)

```julia
julia> sum(val)-tr(A)
-2.4424906541753444e-15
```

### (iii)

```julia
julia> A=Symmetric(A);

julia> val,B=eigen(A);

julia> C = inv(B)*A*B
4×4 Array{Float64,2}:
 -0.392068      1.03273e-16   1.49643e-16   3.13481e-16
  6.75254e-17  -0.00776449    2.61887e-16  -4.62468e-16
  1.54665e-16   4.0726e-16    0.307035     -2.15576e-16
  1.47826e-17  -6.11226e-16  -4.63173e-16   1.91908

julia> norm(inv(B)*A*B - diagm(0 => val))
1.9392390827813702e-15
```

#### observation.

$$
B^{-1}AB=\text{diag}(\lambda_1,...\lambda_n)
$$

where $\lambda_i$ is the eigenvalue of $A$ corresponding to the eigenvector $b_i,$ also being the $i$-th column of $B.$

## 17

```julia
julia> using Plots, FFTW

julia> n = 100; # number of samples

julia> N = rand(n) .- .5; # noise

julia> X = sin.((1:n)*.2); # original signal

julia> Y = N + X; # observed signal

julia> H_f = fft(X) ./ (fft(X) .+ fft(N)) # weiner filter 										in the freq. domain
julia> Y_f = fft(Y);

julia> E = real(ifft(Y_f .* H_f)) # restored signal

julia> plot(Y,label="observed signal");

julia> plot!(X,label="original signal",color=:green);

julia> plot!(E,label="restored signal",color=:red)
```

![WeChat Screenshot_20190719161748](C:\Users\chen1\Desktop\WeChat Screenshot_20190719161748.png)