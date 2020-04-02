# CSC3001 Practice 4

## 2

**Base case.  **$1^3=1=1^2(1+1)^2/4.$

**Induction step.  **Assume $\sum_{i=1}^ki^3=k^2(k+1)^2/4.$ Then $\sum_{i=1}^{k+1}=k^2(k+1)^2/4+k+1=(k+1)^2(k+2)^2/4.\ \ \Box$

## 3

**Base case.  **$\frac11=1\le 1=2-\frac11.$

**Induction step.  **Assume $\sum_{i=1}^k1/i^2\le 2-1/k.$ Then
$$
\sum_{i=1}^{k+1}\frac1{i^2}\le 2-\frac1k+\frac1{(k+1)^2}=2-\frac{(k^2+k+1)/(k^2+k)}{k+1}\le2-\frac{1}{k+1}.\ \ \Box
$$

## 4

**Base case.  **$1-1/2=1/2.$

**Induction step.  **Assume $\prod_{i=2}^k(1-1/i)=1/k.$ Then $\prod_{i=2}^{k+1}(1-1/i)=1/k\cdot k/(k+1)=1/(k+1).\ \ \Box$

## 7

**Base case.  **${0\choose 0}=1=F_{0+1}.$

**(Strong) Induction step.  **Assume $\sum_{i=0}^k{k-i\choose{i}}=F_{k+1}$ for all $k\le n.$ Then
$$
\begin{aligned}
\sum_{i=0}^{n+1}{n+1-i\choose{i}}&=

{n+1\choose 0}+\sum_{i=1}^{n}{n+1-i\choose{i}}
\\&=

{n\choose 0}+\sum_{i=1}^{n}{n+1-i\choose{i}}
\\&=

\sum_{i=1}^{n}{n-i\choose{i-1}}+\sum_{i=0}^{n}{n-i\choose{i}}
\\&=

\sum_{i=0}^{n-1}{n-1-i\choose{i}}+\sum_{i=0}^{n}{n-i\choose{i}}\\

&=F_{(n-1)+1}+F_{(n)+1}\\

&=F_{(n+1)+1}.\ \ \Box
\end{aligned}
$$

## 9

We prove a stronger result: $\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}^n=\begin{bmatrix}
1&2n\\
0&1
\end{bmatrix}$ for all positive integer $n.$

**Base case.  **$\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}^1=\begin{bmatrix}
1&2\cdot 1\\
0&1
\end{bmatrix}$ 

**Induction step.  **Assume $\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}^k=\begin{bmatrix}
1&2k\\
0&1
\end{bmatrix}.$ Then
$$
\begin{aligned}
\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}^{k+1}&=\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}^k\\&=

\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}
\begin{bmatrix}
1&2k\\
0&1
\end{bmatrix}
\\&=

\begin{bmatrix}
1&2(k+1)\\
0&1
\end{bmatrix}.\ \ \Box


\end{aligned}
$$
