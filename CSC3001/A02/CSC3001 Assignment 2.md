# CSC3001 Assignment 2

## Jamie CHEN Ang (118010009)

## 1

BLABLABLAH

## 2

Suppose
$$
A^n=\left(\begin{array}{cc}

a_n & b_n\\
c_n & d_n

\end{array}\right).
$$
Then each entry is given recursively by
$$
a_0=1 ,b_0=0,c_0=0,d_0=1;\\
a_n=a_{n-1}+4b_{n-1},b_{n}=-a_{n-1}+5b_{n-1},\\c_n=c_{n-1}+4d_{n-1},d_n=-c_{n-1}+5d_{n-1}.
$$
Let $a(x),b(x),c(x),d(x)$ be the generating function of $a_n,b_n,c_n,d_n,$ i.e.,
$$
a(x)=\sum_{k\ge0}a_kx^k,b(x)=\sum_{k\ge0}b_kx^k,c(x)=\sum_{k\ge0}c_kx^k,d(x)=\sum_{k\ge0}d_kx^k.
$$
Then
$$
\begin{aligned}
a(x)-1&=\sum_{k\ge1}a_kx^k\\&=\sum_{k\ge1}a_{k-1}x^k+4\sum_{k\ge1}b_{k-1}x^k\\&=
x[a(x)+4b(x)];
\end{aligned}
$$

$$
\begin{aligned}
b(x)&=\sum_{k\ge1}b_kx^k\\&=-\sum_{k\ge1}a_{k-1}x^k+5\sum_{k\ge1}b_{k-1}x^k\\&=
x[-a(x)+5b(x)],
\end{aligned}
$$

yielding
$$
b(x)=\frac{-x}{(3x-1)^2}=\sum_{k\ge0}x^k(-1)(3^{k-1}k);
$$

$$
a(x)=\frac{1-5x}{(3x-1)^2}=\sum_{k\ge0}x^{k}(3^{k}(k+1))-5\sum_{k\ge1}x^k(3^{k-1}k)=\sum_{k\ge0}x^k(3-2k)3^{k-1}.
$$

Therefore $a_n=3^{n-1}(3-2n),b_n=-3^{n-1}n.$ Similarly,
$$
\begin{aligned}
c(x)&=\sum_{k\ge0}c_kx^k\\&=\sum_{k\ge1}c_{k-1}x^k+4\sum_{k\ge1}d_{k-1}x^k\\&=
x[c(x)+4d(x)];
\end{aligned}
$$

$$
\begin{aligned}
d(x)&=\sum_{k\ge0}d_kx^k\\&=1-\sum_{k\ge1}c_{k-1}x^k+5\sum_{k\ge1}d_{k-1}x^k\\&=
1-x[c(x)+5d(x)].
\end{aligned}
$$

Solve for $c(x)$ and $d(x):$
$$
d(x)=\frac{1-x}{(1-3x)^2}=\sum_{k\ge0}x^k(3-2k)3^{k-1}+4x^k(3^{k-1}k)=\sum_{k\ge0}x^k(3+2k)3^{k-1};
$$

$$
c(x)=\frac{4x}{(1-3x)^2}=\sum_{k\ge0}4x^k(3^{k-1}k),
$$

yielding $c_n=3^{n-1}(4n),d_n=3^{n-1}(3+2n).$ Therefore
$$
\boxed{A^n=3^{n-1}\left(\begin{array}{cc}

3-2n & -n\\
4n & 3+2n

\end{array}\right).}
$$

## 3

Modulo $9:$

| $x$   | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $x^3$ | 0    | 1    | -1   | 0    | 1    | -1   | 0    | 1    | -1   |

Therefore $x^3+y^3+z^3\not\equiv 4\pmod4,$ and the equation has no solution.$\ \ \ \Box$

## 4

$$
\begin{aligned}
f(x)&=1\cdot g(x)+\underbrace{x^5-4x^4-6x^3+7x^2-16x+30}_{r_1(x)}\\
g(x)&=x\cdot r_1(x)+\underbrace{4x^5+7x^4-x^3+5x^2-18x-18}_{r_2(x)}\\
r_1(x)&=\frac14\cdot r_2(x)\underbrace{-\frac{23}4(x^4+x^3-x^2+2x-6)}_{r_3(x)}\\
r_2(x)&=\frac{-16x}{23}\cdot r_3(x)+\underbrace{3(x^4+x^3-x^2+2x-6)}_{r_4(x)}\\
r_3(x)&=\frac{-23}{12}r_4(x)+0.
\end{aligned}
$$

Therefore $ s(x):=\text{GCF}[f(x) ,g(x)]=r_4(x)=3(x^4+x^3-x^2+2x-6).$
$$
\begin{aligned}
f(x)&=1\cdot h(x)+\underbrace{x^5-6x^3+3x^2+8x-6}_{R_1(x)}\\
h(x)&=x\cdot R_1(x)+\underbrace{3(x^4+x^3-5x^2-2x+6)}_{R_2(x)}\\
r_1(x)&=\frac x3\cdot R_2(x)\underbrace{-(x^4+x^3-5x^2-2x+6)}_{R_3(x)}\\
r_2(x)&=-3\cdot R_3(x)+0\\
\end{aligned}
$$
Thus, $$t(x):=\text{GCF}[f(x) ,h(x)]=-R_4(x)=x^4 + x^3 - 5 x^2 - 2 x + 6.$$

To find $\text{GCF}[f(x),g(x),h(x)],$ we compute $\text{GCF}[s(x),t(x)]:$
$$
\begin{aligned}
s(x)&=3\cdot t(x)+\underbrace{12(x^2+x-3)}_{l_1(x)}\\
t(x)&=\frac{x^2}{12}\cdot l_1(x)\underbrace{-2(x^2+x-3)}_{l_2(x)}\\
l_1(x)&=-6\cdot l_2(x)+0\\
\end{aligned}
$$
Therefore,
$$
\text{GCF}[f(x),g(x),h(x)]=-l_2(x)=\boxed{2(x^2+x-3).}
$$

## 5

Count from year $2005.$ Let $2005+x$ be any year three comets all achieve perihelion. Then
$$
\begin{cases}
x\equiv6 \pmod{3^15^1}\\
x\equiv6\pmod{2^25^1}\\
x\equiv0 \pmod{2^13^2}
\end{cases}\iff\begin{cases}
x\equiv6 \pmod{3}\\
x\equiv6 \pmod{5}\\
x\equiv6\pmod{2^2}\\
x\equiv6\pmod{5}\\
x\equiv0 \pmod{2}\\
x\equiv0 \pmod{3^2}
\end{cases}\iff\begin{cases}
x\equiv1 \pmod{5}\\
x\equiv2\pmod{4}\\
x\equiv0 \pmod{9}.
\end{cases}
$$
Let $x=9t.$ Then $9t-4s=2$ from the second congruence, yielding $t=2+4k,$ or $x=18+36k.$ Plug in the first congruence, we have $36k-5j=-17,$ yielding $k=-17+5n.$ Hence $x\equiv126 \pmod{180}.$ 

The nearest such year would be the year $(2005+126)=\boxed{2131.}$















## 6

Suffices to show
$$
a=\frac1n,n\ge2
$$
is either a finite decimal or an infinite recurring decimal. 

By FTA, write
$$
n=2^{p}5^qr,
$$
where $p,q\ge0.$ If $r=1, $ then
$$
a=\frac{1}{2^p5^q}=\begin{cases}
{5^{p-q}}\cdot{10^{-p}}\ \ \text{if } p\gt q,\\
{2^{q-p}}\cdot{10^{-q}}\ \ \text{if } q\gt p.
\end{cases}
$$
So $a$ is a finite decimal. 

If $r\neq 1,$ then since $(r,2\cdot5)=1,$ by Euler's Theorem
$$
10^{\phi(r)}\equiv1\pmod {r}.
$$
Then for some $k,$
$$
10^{\phi(r)}-1= kr.\tag1\label1
$$
In this case we write
$$
\begin{aligned}
a&=\frac{1}{2^p5^q}\cdot\frac{1}{r}\\
&=\frac{k}{2^p5^q}\cdot\frac{1}{kr}
\end{aligned}
$$
We already proved $\frac{1}{2^p5^q}$ is a finite decimal, whence $\frac{k}{2^p5^q}$ is also a finite decimal. It suffices to show $\frac{1}{kr}$ is an infinite recurring decimal. Indeed, by $(\ref1),$
$$
\begin{aligned}
\frac{1}{kr}&=\frac{1}{10^{\phi(r)}-1}\\
&= \frac{1}{10^{\phi(r)}}\cdot\frac{1}{1-10^{-\phi(r)}}\\
&= \frac{1}{2^{\phi(r)}5^{\phi(r)}}\cdot\left(1+10^{-\phi(r)}+10^{-2\phi(r)}+\cdots\right).

\end{aligned}
$$
Now, $\frac{1}{2^{\phi(r)}5^{\phi(r)}}$ is, by the proven fact, a finite decimal. And $1+10^{-\phi(r)}+10^{-2\phi(r)}+\cdots$ is infinitely recurring in $\phi(r)$ digits. It follows that $\frac1{kr}$ is an infinite recurring decimal; so is $a.\ \ \ \Box$