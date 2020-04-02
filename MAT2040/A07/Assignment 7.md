# Assignment 7

## Table of contents

[TOC]

## 1

$$
Av_1=[-1,0,1]^T=v_1.
$$

$$
Av_2=[4,-4,4]^T=4v_1.
$$

Hence $v_1,v_2$ are indeed eigenvectors of $A$ with eigenvalues $\lambda_1=1, \lambda_2=4.$

Normalizing $v_1$ and $v_2,$
$$
q_1:=\tilde v_1=[-1,0,1]^T/\sqrt2.
$$

$$
q_2:=\tilde v_2=[1,-1,1]^T/\sqrt3.
$$

Now let
$$
q_3:=q_1\times q_2=[1,2,1]^T/\sqrt6.
$$
We have
$$
Aq_3=[1,2,1]^T/\sqrt6=q_3.
$$
$q_3$ is also an eigenvector of $A$ with eigenvalue $\lambda_3=1.$

Now, since $\{q_1,q_2,q_3\}$ forms an orthonormal basis, by the spectral theorem,
$$
A=Q\Gamma Q^T,
$$
where $Q=[q_1,q_2,q_3],$ and $\Gamma=\text{diag}(\lambda_1,\lambda_2,\lambda_3)=\text{diag}(1,4,1).$

## 2

$$
(B^TAB)^T=B^TA^TB=B^TAB.
$$

$$
(B^TB)^T=B^T(B^T)^T=B^TB.
$$

$$
(BB^T)^T=(B^T)^TB^T=BB^T.
$$

## 3

### (a) 

False.

### (b)

True.

### (c)

False.

### (d)

True.

## 4

$$
x^TAx=\sum_{i,j}a_{ij}x_ix_j=\sum_{i}a_{ii}x_i^2+2\sum_{i<j}a_{ij}x_ix_j.\label1
$$

### (a)

By $(\ref1),$
$$
x^TAx=3x_1^2+2x_2^2+2(2x_1x_2+x_2x_3).\label2
$$

### (b)

Plugging in $(\ref2),$
$$
\begin{aligned}
x^TAx&=12+2+2(2\cdot 2+(-5))\\
&=14+2(-1)\\&=12.
\end{aligned}
$$

### (c)

Plugging in $(\ref2),$
$$
\begin{aligned}x^TAx&=(3+2+2(2+1))/2\\&=(5+6)/2\\&=\frac{11}{2}.
\end{aligned}
$$

## 5

$$
A^TA=\left[
\begin{array}{cc}
9 &-9\\
-9 &9
\end{array}
\right].
$$

Characteristic polynomial of $A^TA$ is
$$
p_{A^TA}(\lambda)=(\lambda-9)^2-81=\lambda(\lambda-18)\implies \lambda_1=18,\lambda_2=0.
$$
Hence the singular values of $A$ are $\sigma_1=3\sqrt2,\sigma_2=0.$

To find the right singular vectors, set
$$
(\lambda I-A^TA)v=0,
$$
yielding
$$
v_{1,2}=[1,\mp1]^T/\sqrt2.
$$
Let
$$
\begin{aligned}u_1&:=Av_1/\sigma_1=[1,-2,-2]^T/3,\\u_2&:=[2,1,0]^T/\sqrt5,\\u_3&:=u_1\times u_2=[2,-4,5]^T/\sqrt{45}.\end{aligned}
$$
Then
$$
A=U\Sigma V^T
$$
where

$\begin{aligned}
&U:=[u_1,u_2,u_3]=\left[
\begin{array}{ccc}
1/3 &2/\sqrt5&2/\sqrt{45}\\
-2/3&1/\sqrt5&-4/\sqrt{45}\\2/3&0&5/\sqrt{45}
\end{array}
\right],
\\
&V^T:=[v_1,v_2]^T=\left[
\begin{array}{ccc}
1/\sqrt2&-1/\sqrt2\\
1/\sqrt2&1/\sqrt2
\end{array}\right],\\
&\Sigma:=\left[
\begin{array}{ccc}
\sigma_1&0\\
0&\sigma_2\\0&0
\end{array}\right]=\left[
\begin{array}{ccc}
3\sqrt2&0\\
0&0\\0&0
\end{array}\right].\end{aligned}$

## 6

Recall that
$$
M \text{ has positive eigenvalues} \iff x^TMx>0\ \forall x.
$$
Therefore $x^TAx,x^TBx >0\ \forall x.$ It immediately follows that
$$
x^T(A+B)x=x^TAx+x^TBx>0+0=0\ \forall x,
$$
which is equivalent to say that $A+B$  has positive eigenvalues.

## 7

Suppose $A$ has eigenvalue decomposition (since $A$ is assumed to be symmetric)
$$
A=Q\Lambda Q^T.
$$
$Q=[q_1,...,q_n]$ has $q_i=$ normalized eigenvectors as columns, and

$\Lambda=\text{diag}(\lambda_1,...,\lambda_n),$ where $\lambda_i$ are eigenvalues corresponding $q_i.$

Now consider the SVD
$$
A=U\Sigma V^T.
$$
$V=[v_1,...v_n]$ where $v_i$ are eigenvectors of $A^TA,$ in this case, $A^2.$ But we already know $A$ and $A^2$ always shares the same eigenvectors. Hence $V=Q.$

$\Sigma=\text{diag}(\sqrt{\tilde\lambda_1},...\sqrt{\tilde\lambda_n}),$ where $\tilde\lambda_i$ are the eigenvalues of $A^TA=A^2,$ which we knows to be $\lambda_i^2:$
$$
\tilde\lambda_i=\lambda_i^2 \ \forall i\label4.
$$
If $A\succeq 0,$ all eigenvalues of $A$ are greater than zero. Taking square roots on both sides of $(\ref4) $ yields
$$
\sqrt{\tilde\lambda_i}=\lambda_i\ \forall i.\label5
$$
Therefore
$$
\Sigma=\text{diag}({\lambda_1},...{\lambda_n})=\Lambda,
$$
which means SVD for a positive-definite matrix is identical to the eigenvalue decomposition:
$$
A=U\Sigma V^T=Q\Lambda Q.
$$
In the case when $A\preceq0, (\ref5)$ becomes
$$
\sqrt{\tilde\lambda_i}=-\lambda_i\ \forall i.
$$

$$
\Sigma=\text{diag}({-\lambda_1},...{-\lambda_n})=-\Lambda.
$$

Hence SVD becomes
$$
A=U\Sigma V^T=(-Q)(-\Lambda)Q^T,
$$
or
$$
A=U\Sigma V^T=Q(-\Lambda)(-Q^T).
$$

## 8

Let $\tilde v=v / \lambda,$ then
$$
\lambda v=Av\implies v=A\tilde v\implies v\in \text{Col }A.
$$

## 9

Using SVD,
$$
A=U\Sigma V^T.
$$

$$
A^T=V\Sigma U^T.
$$

Then
$$
AA^T=U\Sigma V^TV\Sigma U^T=U\Sigma^2U^T.\label6
$$

$$
A^TA=V\Sigma U^TU\Sigma V^T=V\Sigma^2V^T.\label7
$$

Let $Q:=VU^T,$ which is orthogonal. $(\ref6)$ and $(\ref7)$ combined yields
$$
AA^T=U(V^TV)\Sigma^2(V^TV)U^T=UV^T(A^TA)VU^T=Q^T(A^TA)Q,
$$
as desired.

## 10

```julia
julia> using LinearAlgebra

julia> A = [
   1 0 2+2im 0 3-3im
   0 4 0 5 0
   6-6im 0 7 0 8+8im
   0 9 0 1 0
   2+2im 0 3-3im 0 4
   ];
```

### (a)

```julia
julia> H_u = Hermitian(A)
5×5 Hermitian{Complex{Int64},Array{Complex{Int64},2}}:
   1+0im  0+0im  2+2im  0+0im  3-3im
   0+0im  4+0im  0+0im  5+0im  0+0im
   2-2im  0+0im  7+0im  0+0im  8+8im
   0+0im  5+0im  0+0im  1+0im  0+0im
   3+3im  0+0im  8-8im  0+0im  4+0im
  
julia> H_l = Hermitian(A, :L)
5×5 Hermitian{Complex{Int64},Array{Complex{Int64},2}}:
   1+0im  0+0im  6+6im  0+0im  2-2im
   0+0im  4+0im  0+0im  9+0im  0+0im
   6-6im  0+0im  7+0im  0+0im  3+3im
   0+0im  9+0im  0+0im  1+0im  0+0im
   2+2im  0+0im  3-3im  0+0im  4+0im
```

#### observation.

$$
A\neq A^H\implies H_{u}(A)\neq H_{l}(A).
$$

### (b)

```julia
julia> A = Array(Hermitian(rand(Complex{Float64},5,5)));

julia> H_u = Hermitian(A)
5×5 Hermitian{Complex{Float64},Array{Complex{Float64},2}}:
  0.95988+0.0im       0.998716+0.361859im  …  0.772618+0.487604im
 0.998716-0.361859im  0.650059+0.0im          0.661267+0.159806im
 0.579079-0.613569im  0.473588-0.759278im     0.224817+0.274035im
 0.992797-0.45108im    0.12611-0.987278im      0.12303+0.650897im
 0.772618-0.487604im  0.661267-0.159806im     0.386485+0.0im

julia> H_l = Hermitian(A, :L)
5×5 Hermitian{Complex{Float64},Array{Complex{Float64},2}}:
  0.95988+0.0im       0.998716+0.361859im  …  0.772618+0.487604im
 0.998716-0.361859im  0.650059+0.0im          0.661267+0.159806im
 0.579079-0.613569im  0.473588-0.759278im     0.224817+0.274035im
 0.992797-0.45108im    0.12611-0.987278im      0.12303+0.650897im
 0.772618-0.487604im  0.661267-0.159806im     0.386485+0.0im

julia> H_u == H_l
true
```

#### observation.

$$
A= A^H\implies H_{u}(A)= H_{l}(A).
$$

## 11

```julia
julia> A=rand(5,5);

julia> U,S,V=svd(A);

julia> A-U*Diagonal(S)*V'
5×5 Array{Float64,2}:
 -5.55112e-16  -6.66134e-16  …  -3.33067e-16  -4.996e-16
  0.0          -6.245e-16       -3.88578e-16   0.0
 -3.33067e-16  -5.55112e-16     -1.66533e-16  -1.04083e-16
 -3.33067e-16  -2.22045e-16     -1.11022e-16  -1.11022e-16
  0.0          -3.05311e-16     -8.32667e-17   0.0
```

#### observation.

$$
A=USV^T.
$$

## 12

```julia
julia> m = 100; n = 80; p = 120;

julia> A = rand(m, n); B = rand(p, n);

julia> U, V, Q, D1, D2, R0 = svd(A, B);

julia> norm(A - U * D1 * R0 * Q')
7.243315372697871e-13

julia> norm(B - V * D2 * R0 * Q')
8.25995256920901e-13
```

#### observation.

$$
A=UD_1R_0Q^T.
$$

$$
B=VD_2R_0Q^T.
$$

## 13

#### code.

```julia
using Random, StatsBase
Random.seed!(1);

A = Set(['a','e','i','o','u']);
B = Set(['x','y','z']);
omega = 'a':'z';
N = 1e6;

println("mcEst1 \t \tmcEst2")
for _ in 1:5
    mcEst1 = sum([in(sample(omega),A) || in(sample(omega),B) for _ in 1:N])/N
    mcEst2 = sum([in(sample(omega),union(A,B)) for _ in 1:N])/N
    println(mcEst1,"\t",mcEst2)
end
```

#### output.

```julia
mcEst1          mcEst2
0.285158        0.307668
0.285686        0.307815
0.285022        0.308132
0.285357        0.307261
0.285175        0.306606
```

#### analysis.

The estimation given by mcEst2 is correct, as
$$
P(A\cup B)=\frac{5+3}{26}=\frac{4}{13}\approx0.3077.
$$
mcEst1 is a faulty estimator because it provokes *sample* function twice as supposed to once. This means after checking one sample's belongness to $A$ a new sample is drawn for checking belongness to $B.$ mcEst1 really estimates the probability that two i.i.d. discrete RV's, $X_1,X_2\sim \mathcal U(1,26)$ take value $x_1\le5$ or $x_2\le 3,$ namely,
$$
\begin{aligned}
P(x_1\le5 \cup x_2\le3)&=P(x_1\le5)+P(x_2\le3)-P(x_1\le5\cap x_2\le3)\\&=
F_{X_1}(5)+F_{X_2}(3)-F_{X_1}(5)\cdot F_{X_2}(3)\\&=
\frac5{26}+\frac3{26}-\frac{5\cdot3}{26\cdot26}\\&=
\frac4{13}-\frac{15}{676}\\&\approx 0.2855.

\end{aligned}
$$
This coincides with the numerical result.

## 14

#### code.

```julia
N = 1e7; 
a = b = ab = 0;
# a counts for A; b counts for B; ab counts for AB (13)

for _ in 1:N
    tens, ones = divrem(rand(10:25),10)
    if (tens, ones) == (1,3)
        global a += 1; global b += 1; global ab += 1;
    elseif tens == 1
        global a += 1
    elseif ones == 3
        global b += 1
    end
end

fA = a/N; fB = b/N;
fAB = ab/N
fAfB = fA*fB
```

#### output.

```julia
julia> fAB = ab/N
0.0625051

julia> fAfB = fA*fB
0.07811757925852
```

Indeed, the numerical result suggests that $A,B$ are dependent events.