# STA2002 Assignment 1

## 1

Denote $Y:=\sum_{i=1}^n X_i.$ Since $X_i$ are independent RVs, the MGF of $Y$ is given by the product of respective MGFs of $X_i$'s:
$$
M_Y(t)=\prod_{i=1}^nM_{X_i}(t)=\prod_{i=1}^n (1-2t)^{-r_i/2}=(1-2t)^{-\sum_{i=1}^n r_i/2}.
$$
Therefore $Y\sim \chi^2(\sum_{i=1}^n r_i).$

## 2

Let $X_i\sim_{\text{i.i.d.}}\text{Poisson}(1.8)$ be the number of floods within the $i$-th year. Then,
$$
\mu_{X}=1.8\\
\sigma^2_{X}=1.8
$$
Let also $Y_i\sim_{\text{i.i.d.}}\text{Exponential}(\lambda=1/3)$ be the number of days during which ground is flooded, in the span of the $i$'s flood. Then,
$$
\mu_{Y}=3\\
\sigma^2_{Y}=9
$$

### (a)

By CLT,
$$
\frac{\sum_{i=1}^{20}X_i-20\mu_X}{\sqrt{20\sigma_X^2}} \approx Z
$$
where $Z\sim N(0,1),$ the standard normal distribution. Whence
$$
\begin{aligned}
P\left(\sum_{i=1}^{20}X_i\ge 19\right)&\approx P\left(\sqrt{20\sigma_X^2}\cdot Z+20\mu_X\ge 19\right)\\
&=P\left(Z\ge \frac{19-20\mu_X}{\sqrt{20\sigma_X^2}}\right)\\
&=P\left(Z\ge -2.83\right)\\
&= 99.767\%


\end{aligned}
$$

### (b)

By CLT,
$$
\frac{\sum_{i=1}^{120}Y_i-120\mu_Y}{\sqrt{120\sigma_Y^2}} \approx Z
$$
where $Z\sim N(0,1),$ the standard normal distribution. Then,
$$
\begin{aligned}
P\left(\sum_{i=1}^{120}Y_i\lt 365\right)&\approx P\left(\sqrt{120\sigma_Y^2}\cdot Z+120\mu_Y\lt 365\right)\\
&=P\left(Z\lt \frac{365-120\mu_Y}{\sqrt{120\sigma_Y^2}}\right)\\
&=P\left(Z\lt 0.152\right)\\
&= 56.041\%


\end{aligned}
$$

## 3

### (a)

Set sample moment at
$$
\hat\mu_1= \mu_1=\frac1\lambda\implies \hat\lambda_{\text{mom}}=\frac 1{\hat\mu_1}
$$

### (b)

The log-likelihood function is
$$
l_X(\lambda)=\sum_{i=1}^n \ln\left(\lambda e^{-\lambda x_i}\right)=

\sum_{i=1}^n \left[\ln(\lambda)-\lambda x_i\right]=n\ln(\lambda)-\lambda\sum_{i=1}^nx_i
$$
Set the derivative at zero,
$$
l'_X(\lambda)=\frac n\lambda-\sum_{i=1}^n x_i=0\implies \lambda=\frac n {\sum_i x_i}=\frac1{\bar x}
$$
Since $l''_X(\lambda)=-n/\lambda^2<0,$ the point is a global minimum and we conclude
$$
\hat\lambda_{\text{mle}}=\frac1{\bar X}
$$
which coincides with the method of moment estimator.

### (c)

$$
\hat\lambda_{\text{mom}}=\hat\lambda_{\text{mle}}=\frac n {\sum_{i=1}^n X_i}=\frac{6}{18.76}=0.32
$$

### (d)

$$
\mathbb E[\hat\lambda_{\text{mle}}]=\mathbb E[1/\bar X]=n\cdot\mathbb E\left[1/\sum_{i=1}^n X_i\right]=n\cdot\mathbb E[1/G]
$$

where $G:=\sum_{i=1}^nX_i\sim\text{Gamma}(n,\lambda).$ Hence for $n>1,$
$$
\begin{aligned}
\mathbb E[\hat\lambda_{\text{mle}}]&=n\int_{0}^\infty \frac{\lambda^n}{\Gamma(n)}\cdot x^{n-2}e^{-\lambda x}\ dx\\
&=\frac{n\lambda\cdot\Gamma(n-1)}{\Gamma(n)}
\underbrace{\int_{0}^\infty \frac{\lambda^{n-1}}{\Gamma(n-1)}\cdot x^{n-2}e^{-\lambda x}\ dx}_{1}\\
&=\frac{n}{n-1}\lambda\neq \lambda

\end{aligned}
$$
Therefore $\hat\lambda_{\text{mle}}$ is a biased estimator of $\lambda.$ However, it is asymptotically unbiased since $\mathbb E[\hat\lambda_{\text{mle}}]\to \lambda$ as $n\to \infty.$

## 4

Set sample moments
$$
\begin{aligned}
\hat\mu_1 & =\mu_1=\frac{a+b}2\\
\hat\mu_2&=\mu_2=\frac{a^2+ab+b^2}{3}

\end{aligned}\implies
\begin{aligned}
a^2-2\hat\mu_1 a+4\hat\mu_1^2-3\hat\mu_2=0\\
a+b-2\hat\mu_1=0

\end{aligned}
$$
Solving for $a,b$ yields
$$
\hat a_{\text{mom}}=\hat\mu_1-\sqrt{3\left(\hat\mu_2-\hat\mu_1^2\right)} \\
\hat b_{\text{mom}}=\hat\mu_1+\sqrt{3\left(\hat\mu_2-\hat\mu_1^2\right)}
$$

## 5

Denote $m:= \min_{1\le i\le n}\left\{x_i\right\}.$ The likelihood function is given by
$$
L_X(\theta)=\prod_{i=1}^nf_X(x;\theta)=\begin{cases}
\begin{aligned}
\exp\left[\sum_{i=1}^n(\theta-x_i) \right], \quad &\text{if }m\ge\theta\\
0,\quad&\text{otherwise}
\end{aligned}
\end{cases}
$$
If $m\ge\theta,$ we have $L_X(\theta)\gt0,$ and so the MLE can be found by maximizing the log-likelihood function
$$
\begin{aligned}
\theta^*&={\arg\max}_{\theta\le m}\ L_X(\theta)\\
&={\arg\max}_{\theta\le m}\ l_X(\theta)\\
&={\arg\min}_{\theta\le m}\ \sum_{i=1}^n(x_i-\theta)\ge0\\
\end{aligned}
$$
since $\theta\le m\le x_i$ for all $i.$ Moreover, $\theta^*=m$ brings the sum to zero and so must be the minimizer. Hence the MLE in this case is given by
$$
\hat\theta=\min_{1\le i\le n}\left\{X_i\right\}
$$
Otherwise, the likelihood function is constant zero and we may choose $\hat\theta$ to be any RV. Therefore $\hat\theta=\min_{1\le i\le n}\left\{X_i\right\}$ is the final MLE. 

## 6

We first derive the CDF of $\hat\theta$ by noting that
$$
\begin{aligned}
F_\hat\theta(x)&=P\left(\hat\theta\le x\right)\\&=P\left(\max_i\left\{X_i\right\}\le x\right)\\
&=P\left[\cap _i({X_i\le x})\right]\\


\end{aligned}
$$
Since $X_i$ are independently distributed this further equates to
$$
\begin{aligned}
F_\hat\theta(x)&=\prod _i P(X_i\le x)\\
&= \prod_i F_{X_i}(x)\\
&= \prod_i \frac x\theta\\
&=\frac {x^n}{\theta^n}
\end{aligned}
$$
Differentiating $F_\hat\theta(x)$ yields the PDF
$$
f_{\hat\theta}(x)=\frac{d}{dx}F_{\hat\theta}(x)=\frac n {\theta^n}x^{n-1}
$$
Hence the mean of $\hat\theta$ is given by
$$
\begin{aligned}
\mathbb E[\hat\theta]&=\int_{x=0}^\theta xf_{\hat\theta}(x)\ dx\\
&=\int_{x=0}^\theta\frac n {\theta^n}x^{n}\\
&=\left.\frac n {(n+1)\theta^n}x^{n+1}\right|_{x=0}^\theta\\
&=\frac n{n+1}\theta\neq \theta


\end{aligned}
$$
from which we see $\hat\theta$ is biased. However, $\Theta:=(n+1)\hat\theta/n$ would be an unbiased estimator of $\theta$, since
$$
\mathbb E[\Theta]=\frac{n+1}n\mathbb E[\hat\theta]=\theta
$$

## 7

### (a)

We have the conditional probabilities
$$
f_{X|K}(x|k)=\phi(x;\mu_k,\sigma_k^2)\\
$$
Thus the joint PDF is given by
$$
\begin{aligned}
f_{X,K}(x,k)&=P(K=k)\cdot f_{X|K}(x|k)
\\
&=\pi_k \phi\left(x;\mu_k,\sigma_k^2\right)\\
&=\frac{\pi_k}{\sqrt{2\pi\sigma_k^2}}\exp\left[-\frac12\left(\frac{x-\mu_k}{\sigma_k}\right)^2\right]
\end{aligned}
$$
with support $(x,k)\in\R\times\{0,1\}.$

### (b)

Denote the observed values
$$
\begin{aligned}
&i_0:=\{i:k_i=0\},\ i_1:=\{i:k_i=1\}\\
&n_0:=|i_0|,\ n_1=|i_1|\\
&s_0:=\sum_{i\in I_0}x_i,\ s_1:=\sum_{j\in I_1}x_j\\
&q_0:=\sum_{i\in I_0}x_i^2,\ q_1:=\sum_{j\in I_1}x_j^2

\end{aligned}
$$
and the random variables
$$
\begin{aligned}
&I_0:=\{i:K_i=0\},\ I_1:=\{i:K_i=1\}\\
&N_0:=|I_0|,\ N_1=|I_1|\\

&S_0:=\sum_{i\in I_0}X_i,\ S_1:=\sum_{j\in I_1}X_j\\
&Q_0:=\sum_{i\in I_0}X_i^2,\ Q_1:=\sum_{j\in I_1}X_j^2

\end{aligned}
$$
The log-likelihood function of $(X,K)$ is given by
$$
\begin{aligned}
l_{X,K}(\pi_0,\mu_0,\sigma_0^2,\mu_1,\sigma_1^2)&= \sum_{i=1}^n\ln\left[\pi_{k_i}\phi\left(x_n;\mu_{k_i},\sigma_{k_i}^2\right)\right]\\
&=\sum_{i=1}^n\left[-\ln\left({\sqrt{2\pi}}\right)+\ln\left({\pi_{k_i}}/\sigma_{k_i}\right)-\frac12\left(\frac{x_i-\mu_{k_i}}{\sigma_{k_i}}\right)^2\right]\\
&=-n\ln\left({\sqrt{2\pi}}\right)+n_0(\ln{\pi_{0}}-\ln\sigma_{0})-
\frac1{2\sigma_0^2}(q_0-2\mu_0s_0+n_0\mu_0^2)

\\&\qquad\qquad\qquad\ \ +n_1(\ln{\pi_{1}}-\ln\sigma_{1})-
\frac1{2\sigma_1^2}(q_1-2\mu_1s_1+n_1\mu_1^2)

\end{aligned}
$$
Setting gradient at zero,
$$
\nabla l_{X,K}=\left[\begin{array}{c}

{n_0}/{\pi_0}-{n_1}/({1-\pi_0})\\
\frac{1}{\sigma_0^2}(s_0-n_0\mu_0)\\

-\frac{1}2n_0/\sigma_0^2+\frac12\left(\sigma_0^2\right)^{-2}(q_0-2\mu_0s_0+n_0\mu_0^2)\\

\frac{1}{\sigma_1^2}(s_1-n_1\mu_1)\\

-\frac12n_1/\sigma_1^2+\frac12\left(\sigma_1^2\right)^{-2}(q_1-2\mu_1s_1+n_1\mu_1^2)
\end{array}\right]=0
$$
we obtain
$$
\begin{cases}
\pi_0=n_0/(n_0+n_1)\\
\mu_0=s_0/n_0\\
\sigma_0^2=q_0/n_0-(s_0/n_0)^2\\
\mu_1=s_1/n_1\\
\sigma_1^2=q_1/n_1-(s_1/n_1)^2
\end{cases}
$$
from which we deduce the MLE's
$$
\begin{cases}
\hat\pi_0=N_0/(N_0+N_1)\\
\hat\mu_0=S_0/N_0\\
\hat\sigma_0^2=Q_0/N_0-(S_0/N_0)^2\\
\hat\mu_1=S_1/N_1\\
\hat\sigma_1^2=Q_1/N_1-(S_1/N_1)^2
\end{cases}
$$

### (c)

Python code:

```python
import csv
import os, sys

with open(os.path.join(sys.path[0], 'GMM.csv'), newline='') as f:
    reader = csv.reader(f)
    lst = list(reader)[1::]

n_0 = n_1 = s_0 = s_1 = q_0 = q_1 = 0
for r in lst:
    k, x = int(r[0]), float(r[1])
    if k:
        n_1 += 1
        s_1 += x
        q_1 += x**2
    else:
        n_0 += 1
        s_0 += x
        q_0 += x**2

pi_0  = n_0 / (n_0 + n_1)
mu_0  = s_0 / n_0
var_0 = q_0 / n_0 - (s_0 / n_0) ** 2
mu_1  = s_1 / n_1
var_1 = q_1 / n_1 - (s_1 / n_1) ** 2

print('pi_0  = ', pi_0, '\n',
        'mu_0  = ', mu_0, '\n',
        'var_0 = ', var_0, '\n',
        'mu_1  = ', mu_1, '\n', 
        'var_1 = ', var_1, sep='')
```

Output:

```python
pi_0  = 0.94914
mu_0  = 49.95766742434471
var_0 = 99.8605412030156
mu_1  = 60.81269474517893
var_1 = 101.61414449594167
```
