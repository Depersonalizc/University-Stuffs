# CSC3001 Practice Exercise 3

## Direct proof

### 4

Denote the product by $\prod.$ Any five consecutive integers $n+1,...,n+5$ must contain one that's divisible by $5.$ Since, by division algorithm $n+5=5q+r$ with $0\le r\lt5.$ If $r=0$ we are done. Otherwise $5q+r-k=5q$ with $0\lt k\lt5;\ n+r=5q$ with $1 \le r\le4.$ Similarly one must be divisible by $4,$ one by $3,$ one by $2.$ Therefore $3,5|\prod.$ Since $(3,5)=1$ it follows that their product $ 15|\prod.$ Also note that if $4|a,$ then $2|a,$ then $2|a-2,a+2.$ So two distinctive integers are divisible by $2$ and by $4$ respectively. Hence $8|\prod.$ But $(8,15)=1.$ Therefore $120|\prod.\ \ \Box$

### 5

If $n=2k+1,n^2-1=4k(k+1).$ Note that either $k$ or $k+1$ is even. Hence $8|(n^2-1).$

## Proof by contrapositive

### 5

Suppose $n$ is composite. Then $n=ab$ with both $a,b>1.$ Then $2^n-1=(2^a)^b-1=(a-1)\sum_{i=0}^{b-1}2^{ai}.$ Hence $2^n-1$ is composite.$\ \ \Box$

## Proof by contradiction

### 6

Suppose $a^2-4b-3=0.$ Then $2|4|(a^2+1).$ Then $2|(a^2-1)=(a+1)(a-1).$ Therefore $a$ is odd. Let $a=2k+1.$ From above $4|(a^2+1)=(4k^2+4k+2),$ which implies $4|2.$ Contradiction. $\ \ \Box$

## Proof by cases

### 2

$$
\max\{x,y\}+\min\{x,y\}=\begin{cases}
x+y\ \ \ \text{ if }x\ge y\\
y+x\ \ \ \text{ if }x\lt y
\end{cases}=x+y.\ \ \Box
$$

