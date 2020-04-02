# CSC3001 Assignment 1

## 1

Suppose not, then $S:=\{2^{4n-1}:n\in\Z^+,2^{4n-1}\not\equiv 8 \pmod {10}\}$ is a non-empty subset of $\N.$ By Well Ordering Principle $S$ has a minimum $m,$ with $m=2^{4k-1}$ for some integer $k\ge 2.$ But then $m/16=2^{4(k-1)-1}\not\equiv 8\pmod{10}.$ For, if $m/16\equiv 8\pmod{10},$ $m\equiv 8\cdot 16=128\equiv 8 \pmod {10}.$ This implies that $m/16\in S,$ contradicting the minimality of $m.$ Therefore $2^{4n-1}\equiv 8\pmod{10}$ for all $n\in \Z^+.\ \ \Box$

## 2

For fixed $x\ge 2 ,$ we proceed by induction on $y.$ Clearly $x^2\ge x+2.$ Assume $x^k\ge x+k$ for some $k\ge2.$ Then $x^{k+1}\ge (x+k)x=x^2+kx\ge x+kx+2\ge x+k.$ So $\forall y\ge2(x^y\ge x+y).$ By Universal Generalization $\forall x\ge2(\forall y\ge2(x^y\ge x+y)).\ \ \Box$

## 3

We first prove $ \cup_{r\in\Z}[r]=\Z[x].$

If $y\in [r]$ for some integer $r,$ then $y=(x-1)p(x)+r$ for some $p(x)\in \Z[x].$ Therefore
$$
\begin{aligned}
y&=r+(x-1)\sum_{i=0}^na_ix^i\\&=r+\sum_{i=0}^na_ix^{i+1}-a_ix^i\\&=
(r-a_0)x^0+\sum_{i=1}^n(a_{i-1}-a_i)x^i+a_nx^{n+1}\\&=
\sum_{i=0}^{n+1}b_ix^i\in\Z[x].
\end{aligned}\tag1\label1
$$
Conversely, if $y=\sum_{i=0}^nb_ix_i,$ pick integers $r$ and $a_0$ s.t. $b_0=r-a_0.$ Pick $a_1=a_0-b_1, a_2=a_1-b_2=a_0-(b_1+b_2),...,a_{n-1}=a_0-\sum_{i=1}^{n-1} b_i.$ Reverse the algebra in $(\ref1),$ we have $y=r+(x-1)\sum_{i=0}^{n-1}a_ix_i\in [r].$ So $\cup_{r\in\Z}[r]=\Z[x].$

Next we prove $[s]\cap[r]=\empty$ whenever $s\neq r.$ 

Let $s\neq r.$ Suppose there exists $y\in [s]\cap [r]$ with
$$
y=(x-1)p(x)+s=(x-1)q(x)+r  \tag2  \label3
$$
for some $p(x),q(x)\in \Z[x].$ Let $n:=\max \{\deg(p),\deg(q)\},$ where $\deg(\cdot)$ denotes the degree of a polynomial. From $( \ref3),$
$$
\begin{aligned}
r-s&=
\sum_{i=0}^n(p_i-q_i)(x-1)x^i\\
&=(p_0-q_0)(-1)+\sum_{i=1}^n[(p_{i-1}-q_{i-1})-(p_i-q_i)]x_i+(p_n-q_n)x^{n+1}.

\end{aligned}
$$
Therefore $p_n-q_n=p_{n-1}-q_{n-1}=...=p_0-q_0=0.$ But then $r-s=0,$ a contradiction. Therefore $[s]\cap [r]=\empty$ given $s\neq r.$ This completes the proof.$\ \ \Box$ 

## 4

### (i)

$$
\forall a[\neg(a\in\Z^+)\or (\forall k[\neg(k\in\Z^+)\or n\neq (k+2)(a+\frac{k+1}{2})])].
$$

### (ii)

Suppose $S(n)$ is false, that is, $n=(k+2)(a+\frac{k+1}{2})$ for some positive integers $a, k.$ Then $(k+2)|n,n$ is not prime.

Suppose $n$ is not prime, then by Prime Factorization $n=pq$ for some positive integers $p,q,$ where $p$ is a prime greater than $2$ ($n$ is odd) and $q>1. $ ($n$ is not prime itself) Pick $k:=p-2\ge1.$ Since $p$ is odd $k+1=p-1$ is even. We may then pick integer $a:=q-\frac{k+1}2\ge q-1\ge1.$ Now $(k+2)(a+\frac{k+1}{2})=pq=n$ with both $a,k\in \Z^ +.$ $S(n)$ is false. $\ \ \Box$

## 5

No.

For $n=3,$ note that each dicube contributes either $3$ (if one of the unit cube in the dicube occupies a corner) or $5$ (if the dicube does not occupy the corner) to the area of exterior surface of the $3\times3$ hull. If the hull can be tiled, assume $x$ dicubes contribute $3,$ $y$ dicubes contribute $5.$ Then we have a Diophantine system:
$$
\begin{cases}
3x+5y=A_\text{ext}=54\\
2x+2y=V_{\text{hull}}=26.
\end{cases}
$$
This system has no solution in $\N^2.$ Therefore the hull cannot always be tiled.$\  \ \Box$

## 6

Suppose $n$ days. The total money spent is
$$
(x+y+z)n=15+17+19=51.
$$
Note that $51$ has $3$ and $17$ as its unique prime factors. Since $x\lt y\lt z,$ we conclude that $x+y+z=17$ and $n=3.$ Since Cony spent $17$ in total, he must have bought distinctive snacks on each day (cannot all be the same item, as $3\not|\ \ 17;$ cannot be that exactly two are the same, as $2a+b\neq x+y+z=17,\forall a,b\in\{x,y,z\},a\neq b$). WLOG assume Brown spent $y$ on day 1. Then Cony spent $y$ either on day 2 or day 3.  WLOG assume day 2. Then Brown couldn't have spent $y$ on day 2. He couldn't have spent $z$ either, as otherwise he would've spent at least $y+z+x=17.$ So Brown spent $x$ on day 2. Then Sally spent $z$ on day 2. So far three possibilities remain:

|       | brown    | cony  | sally    | brown |  cony | sally |      |
| ----- | :------- | :---- | :------- | ----: | ----: | ----: | ---- |
| D1    | $y$      | $x$   | $z$      |   $y$ |   $z$ |   $x$ |      |
| D2    | $x$      | $y$   | $z$      |   $x$ |   $y$ |   $z$ |      |
| D3    | $x\ (y)$ | $z$   | $y\ (x)$ |   $y$ |   $x$ |   $z$ |      |
| Total | $15$     | $ 17$ | $19$     | $15 $ | $17 $ | $19 $ |      |

which correspond to two Diophantine systems:
$$
\begin{cases}
2x+y=15
\\
x+y+z=17\\
y+2z=19
\end{cases}
\ \ \ \ \ \ \ \ \ \ \ \ \ 
\begin{cases}
x+2y=15
\\
x+y+z=17\\
x+2z=19
\end{cases}
$$
each yielding
$$
\begin{cases}
x=t\\
y=15-2t\\
z=t+2
\end{cases}
\ \ \ \ \ \ \ \ \ \ \ \ \ 
\begin{cases}
x=15-2t\\
y=t\\
z=t+2
\end{cases},t\in\Z.
$$
Considering $0<x<y<z,$ the first system yields no solution. From the second system, $0<15-2t<t<t+2,$ equivalently $5<t<7.5;t=6$ or $7.$ All solutions are thus given by ordered pairs $(x,y,z)=(3,6,8),(1,7,9).$

