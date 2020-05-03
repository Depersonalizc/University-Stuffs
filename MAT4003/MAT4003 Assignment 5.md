# MAT4003 Assignment 5

## 1

### (a)

By the Reciprocity Law,
$$
\left(\frac{3}{p}\right)=\begin{cases}
\left(\frac{p}{3}\right)\text{, if }p\equiv1\pmod{4}\\
-\left(\frac{p}{3}\right)\text{, if }p\equiv3\pmod{4}

\end{cases}.
$$
Now
$$
\left(\frac{p}{3}\right)=\begin{cases}
\left(\frac{1}{3}\right)=1\text{, if }p\equiv1\pmod{3}\\
\left(\frac{2}{3}\right)=-1\text{, if }p\equiv2\pmod{3}

\end{cases}.
$$
Therefore
$$
\left(\frac{3}{p}\right)=1\iff\left\{ p\equiv\begin{cases}
1\pmod 4\\1\pmod 3
\end{cases}\or 
p\equiv\begin{cases}
3\pmod 4\\2\pmod 3
\end{cases}\right\},
$$
equivalently by CRT
$$
p\equiv\pm1\pmod{12}.
$$

### (b)

$$
\left(\frac{-3}{p}\right)=\left(\frac{-1}{p}\right)\left(\frac{3}{p}\right)=\begin{cases}
\left(\frac{3}{p}\right)\text{, if }p\equiv1\pmod{4}\\
-\left(\frac{3}{p}\right)\text{, if }p\equiv3\pmod{4}

\end{cases}.
$$

Therefore
$$
\left(\frac{-3}{p}\right)=1\iff\left\{ p\equiv\begin{cases}
1\pmod 4\\\pm1\pmod{12}
\end{cases}\or 
\begin{cases}
p\equiv3\pmod 4\\p\not\equiv\pm1\pmod{12}
\end{cases}\right\}.
$$
Equivalently,
$$
p\equiv 1,\text{ or }3,\text{ or }7\pmod {12}.
$$

## 2

### (a)

$$
\left(\frac{27}{101}\right)=\left(\frac{101}{27}\right)=\left(\frac{20}{27}\right)=\left(\frac{5}{27}\right)=\left(\frac{27}{5}\right)=\left(\frac{2}{5}\right)=-1
$$

### (b)

$$
\left(\frac{111}{1001}\right)=\left(\frac{1001}{111}\right)=\left(\frac{2}{111}\right)=1
$$

### (c)

$$
\left(\frac{1009}{2307}\right)=\left(\frac{2307}{1009}\right)=\left(\frac{289}{1009}\right)=\left(\frac{142}{289}\right)=\left(\frac{71}{289}\right)=\left(\frac{5}{71}\right)=\left(\frac{1}{5}\right)=1
$$

## 3

### (a)

By CRT,
$$
x^2\equiv-2\pmod{118}\iff \begin{cases}
x^2\equiv-2\pmod2\\
x^2\equiv-2\pmod{59}
\end{cases}\iff \begin{cases}
x\equiv0\pmod2\\
x^2\equiv-2\pmod{59}
\end{cases}.
$$
Since
$$
\left(\frac{-2}{59}\right)=\left(\frac{-1}{59}\right)\left(\frac{2}{59}\right)=(-1)^2=1,
$$
$-2$ is a quadratic residue modulo $59.$ Further, Theorem $(4.19)$ asserts that there are exactly $(2,\phi(59))=2$ solutions to the congruence $x^2\equiv-2\pmod{59}.$ Again by CRT, there are exactly $1\cdot 2=2$ solutions to the original congruence.

### (b)

$$
x^2\equiv-1\pmod{244}\iff \begin{cases}
x^2\equiv-1\pmod{4}\\
x^2\equiv-1\pmod{61}
\end{cases}.
$$

Since $x^2\equiv 0,1,0,1\not\equiv-1\pmod4$ for each $x\equiv 0,1,2,3\pmod 4,$ there are no solutions to the original congruence.

### (c)

$$
x^2\equiv-1\pmod{365}\iff \begin{cases}
x^2\equiv-1\pmod{5} \ \ \ (*)\\
x^2\equiv-1\pmod{73}\ \ \ (**)
\end{cases}.
$$

Since
$$
\left(\frac{-1}{5}\right)=\left(\frac{-1}{73}\right)=1,
$$
$-1$ is a quadratic residue moduli $5$ and $73.$ It follows from Theorem $(4.19)$ there are exactly $2$ solutions, each for $(*)$ and $(**).$ Combined with CRT, there are exactly $2^2=4$ solutions to the original congruence.

### (d)

Note that $227$ is a prime. By the Reciprocity Law,
$$
\left(\frac{7}{227}\right)=-\left(\frac{3}{7}\right)=\left(\frac{1}{3}\right)=1,
$$
$7$ is a quadratic residue modulo $227.$ By Theorem $(4.19)$ there are exactly two solutions.

### (e)

$$
x^2\equiv267\pmod{789}\iff \begin{cases}
x^2\equiv0\pmod{3}\\
x^2\equiv4\pmod{263}
\end{cases}\iff \begin{cases}
x\equiv0\pmod{3}\\
x^2\equiv4\pmod{263}
\end{cases}.
$$

Now $\left(\frac{4}{263}\right)=1.$ Therefore by Theorem $(4.19)$ there are exactly two solutions to the congruence $x^2\equiv4\pmod{263}.$ By CRT there are exactly $1\cdot 2=2$ solutions to the original congruence.

### (f)

Let $y:=x^2.$ Then $y^2\equiv 25\pmod{1013}.$ By Theorem $(4.19)$ $y\equiv\pm5\pmod{1013}$ are the only two solutions to the congruence $y^2\equiv 25\pmod{1013}.$ Hence $x^2\equiv \pm5\pmod{1013}.$ Now by the Reciprocity Law,
$$
\left(\frac{5}{1013}\right)=\left(\frac{3}{5}\right)=\left(\frac{-1}{3}\right)=-1
$$

$$
\left(\frac{-5}{1013}\right)=-\left(\frac{-1}{1013}\right)=-1.
$$

Therefore no solutions to the congruence exist.

## 4

$$
x^2\equiv a\pmod{n}\iff \begin{cases}
x^2\equiv a\pmod{p_1^{\alpha_1}}\\
x^2\equiv a\pmod{p_2^{\alpha_2}} \\
\vdots\\
x^2\equiv a\pmod{p_r^{\alpha_r}} \\
\end{cases}.
$$

Consider
$$
x^2\equiv a\pmod{p_i^{\alpha_i}}.\tag1
$$
Since $a$ and $n$ are coprime, so are $a$ and $p_i^{\alpha_i}.$ Then by Theorem $(5.9)$ solutions exist if and only if $\left(\frac{a}{p_i}\right)=1. $ Combining Theorem $(4.19),$ the number of solutions to $(1) $ is
$$
\begin{cases}
2\text{, if }\left(\frac{a}{p_i}\right)=1\\
0\text{, if }\left(\frac{a}{p_i}\right)=-1\\

\end{cases}=\left(\frac{a}{p_i}\right)+1.
$$
Hence the number of solutions to $x^2\equiv a\pmod{n}$ is
$$
\prod_{i=1}^r\left[\left(\frac{a}{p_i}\right)+1\right].
$$

## 5

By Reciprocity Law for Jacobi Symbols,
$$
\left(\frac{15}{n}\right)=\begin{cases}
\left(\frac{n}{15}\right)\text{, if }n\equiv1\pmod4\\
-\left(\frac{n}{15}\right)\text{, if }n\equiv3\pmod4
\end{cases}.
$$
If $(n,15)>1,$ then some factor of $n$ divides $15,$ we have $\left(\frac{15}{n}\right)=0. $ Hence it suffices to investigate $n\in\Z_{15}^*:$

| $n$                         | 1    | 2    | 4    | 7     | 8    | 11     | 13     | 14     |
| --------------------------- | ---- | ---- | ---- | ----- | ---- | ------ | ------ | ------ |
| $\left(\frac{n}{15}\right)$ | $1$  | $1$  | $1$  | $-1 $ | $1$  | $-1  $ | $-1  $ | $-1  $ |

Therefore
$$
\left(\frac{15}{n}\right)=1\iff n\equiv\begin{cases}
1\pmod4\\
1,2,4,8\pmod{15}
\end{cases}\text{or}\begin{cases}
3\pmod4\\
7,11,13,14\pmod{15}
\end{cases}.
$$
By CRT,
$$
n\equiv1,7,11,17,43,49,53,\text{or }59\pmod{60}.
$$

## 6

By CRT,
$$
x^2\equiv a\pmod{21}\iff \begin{cases}
x^2\equiv a\pmod{3}\\
x^2\equiv a\pmod{7}
\end{cases}.
$$
We shall find all $a\in\Z^*_{21}$ s.t.
$$
\left(\frac{a}{3}\right)=\left(\frac{a}{7}\right)=-1.
$$
Clearly,
$$
a\equiv2\pmod 3.\tag1
$$
Also, since $1,2,4$ are quadratic residues modulo $7,$ as
$$
1^2\equiv1;5^2\equiv25\equiv4;3^2\equiv9\equiv2,
$$
the remaining $3,5,6$ are all the quadratic non-residues modulo $7,$ by Remark on Page $5$ of Section $5.$ Hence
$$
a\equiv3,5\text{, or }6\pmod 7.\tag2
$$
Combining $(1)$ and $(2)$ by CRT, we have
$$
a\equiv5,17\text{, or }20\pmod {21}.
$$

## 7

$$
\sum_{j=1}^{p-2}\left(\frac{j(j+1)}{p}\right)=\sum_{j=1}^{p-2}\left[\left(\frac{j^2}{p}\right)\left(\frac{j^{-1}+1}{p}\right)\right]=\sum_{j=2}^{p-1}\left(\frac{j^{-1}}{p}\right)=\sum_{j=2}^{p-1}\left(\frac{j}{p}\right)
$$

Now, since there are exactly $\frac{p-1}2$ quadratic residues and $\frac{p-1}2$ non-residues in $\Z_p^*,$ it follows that
$$
\sum_{j=1}^{p-2}\left(\frac{j(j+1)}{p}\right)=\left[\sum_{j=1}^{p-1}\left(\frac{j}{p}\right)\right]-\left(\frac{1}{p}\right)=0-1=-1.
$$

## 8

Write $n=p_1^{a_1}\cdots p_r^{a_r}.$ Since $n$ is not a perfect square,
$$
\left(\frac{k}{n}\right)=\prod_{i=1}^r\left(\frac{k}{p_i}\right)^{a_i}=\prod_{i=1}^{s}\left(\frac{k}{q_i}\right)^{b_i}.
$$
where $q_1,\cdots,q_s$ are all prime factors of $n$ with odd exponents $b_i.$ By Remark on Page $5$ of Section $5,$ exactly half of the $k_j$ in $\Z_{q_i}^*$ are quadratic residues, correspondingly $\left(\frac{k_j}{q_i}\right)^{a_i}=-1,$ while the rest are non-residues, with $\left(\frac{k_j}{q_i}\right)^{a_i}=1.$  Denote $R_i:=\{k:k\in \Z_{q_i}^*,\left(\frac{k}{q_i}\right)=1\}$ being the reduced quadratic residues modulo $q_i,$ then
$$
\begin{aligned}
\left(\frac{k}{n}\right)=1&\iff\left(\frac{k}{q_i}\right)^{b_i}=-1\text{ for even number of } i\text{'s}.\\
&\iff k\equiv r\in R_i\pmod{q_i}\text{ for even number of } i\text{'s}
\end{aligned}
$$
which corresponds to half of the systems of congruence in$\Z_{p_1}^*\times\Z_{p_2}^*\times\cdots\times\Z_{p_s}.$ Hence there are a half of the $k\in\Z_n^*$ with $\left(\frac{k}{n}\right)=1.$ The rest are $k$ with $\left(\frac{k}{n}\right)=-1.$

## 9

$$
S_1=\{x:x\in\Z_p^*,\left(\frac{x}{p}\right)=1\}\\S_2=\{x:x\in\Z_p^*,\left(\frac{x}{p}\right)=-1\}
$$

## 10

### (a)

Consider $p:=1+k\cdot8q_1q_2\cdots q_n,$ where $q_1=2,q_2,\cdots,q_n$ are all the primes $\le M.$  Given $N\in[-M,M], $ and $N$ coprime to $p, $ we may then write $N=\pm 2^{a_1}q_2^{a_2}\cdots q_n^{a_n}.$ Since $p\equiv 1\pmod4,$ we have
$$
\left(\frac{N}{p}\right)=\left(\frac{\pm1}{p}\right)\left(\frac{2^{a_1}q_2^{a_2}\cdots q_n^{a_n}}{p}\right)=\left(\frac{2^{a_1}q_2^{a_2}\cdots q_n^{a_n}}{p}\right).
$$
Since $p\equiv1\pmod 8 $ by definition, $\left(\frac{2}{p}\right)=1,$ hence
$$
\left(\frac{2^{a_1}q_2^{a_2}\cdots q_n^{a_n}}{p}\right)=\left(\frac{q_2^{a_2}\cdots q_n^{a_n}}{p}\right).
$$
By the Reciprocity Law, and that $p\equiv1\pmod4,$
$$
\left(\frac{q_2^{a_2}\cdots q_n^{a_n}}{p}\right)=\prod_{i=2}^{n}\left(\frac{q_i}{p}\right)^{a_i}=\prod_{i=2}^{n}\left(\frac{p}{q_i}\right)^{a_i}=\prod_{i=2}^{n}\left(\frac{1}{q_i}\right)^{a_i}=1.
$$
Therefore
$$
\left(\frac{N}{p}\right)=1.
$$

### (b)

Let $g$ be *any* primitive root modulo $p,$ then by Remark on Page $5$ of Section $5,$
$$
g^2,g^4,\cdots,g^{p-1}
$$
are all the quadratic residues modulo $p. $ Therefore any quadratic residue modulo $p$ cannot be a primitive root modulo $p.$ Specifically, all $N$ in $(a)$ cannot be primitive roots modulo $p. $ 

### (c)

Following $(b),$ all integers $N\in[0,M]\cup[p-M,p]$ that are coprime to $p$ cannot be primitive roots modulo $p.$ Let $g\in\Z_p^*$ be any primitive root modulo $p.$ $g$ is necessarily coprime to $p,$  implying $g\not\in[0,M]\cup[p-M,p].$ So $g\in(M,p-M),$ whence the minimum $r_p\in(M,p-M).$