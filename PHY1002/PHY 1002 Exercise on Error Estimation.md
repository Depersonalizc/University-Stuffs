# PHY 1002: Exercise on Error Estimation

## 1

| $1234$     | $123400$              | $123.4$                | $1001$      | $1000.$                 | $10.10$                  | $0.0001010$ | $100.0$ |
| ---------- | --------------------- | ---------------------- | ----------- | ----------------------- | ------------------------ | ----------- | ------- |
| 4          | 4                     | 4                      | 4           | 4                       | 4                        | 4           | 4       |
| **$1010$** | **$1.01\times 10^3$** | **$1.010\times 10^3$** | **$0.015$** | **$1.5\times 10^{-2}$** | **$1.50\times 10^{-2}$** |             |         |
| 3          | 3                     | 4                      | 2           | 2                       | 3                        |             |         |

## 2

Expand $f$ at point $(\bar u, \bar v)$ to the first order:
$$
f(u_i,v_i)= f(\bar u,\bar v)+f_u(\bar u)\cdot(u_i-\bar u)+f_v(\bar v)\cdot(v_i-\bar v)+...
$$
Therefore
$$
\begin{aligned}
\sigma^2_x&=\lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N(x_i-\bar x)^2\right]\\
&= \lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N\left(f(u_i,v_i)-{f(\bar u,\bar v)}\right)^2\right]\\
&\approx\lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N\left(f_u(\bar u,\bar v)\cdot(u_i-\bar u)+f_v(\bar u,\bar v)\cdot(v_i-\bar v)\right)^2\right]\\
&=\lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N f_u^2\cdot(u_i-\bar u)^2+f_v^2\cdot(v_i-\bar v)^2+2f_u f_v\cdot(u_i-\bar u)(v_i-\bar v)\right]_{(\bar u,\bar v)}\\
&=\lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N (u_i-\bar u)^2\right]f_u^2(\bar u,\bar v)+\lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N (v_i-\bar v)^2\right]f_v^2(\bar u,\bar v)+\lim_{N\to\infty}\left[\frac1N\sum_{i=1}^N (u_i-\bar u)(v_i-\bar v)\right]f_u\cdot f_v(\bar u,\bar v)\\
&=[\sigma_u^2f_u^2+\sigma_v^2f_v^2+2\sigma_{uv}^2f_u f_v]_{(\bar u,\bar v)}.
\end{aligned}
$$
If $\text{Cov}(u,v)=0,$ then we have
$$
\sigma_x^2\approx [\sigma_u^2f_u^2+\sigma_v^2f_v^2]_{(\bar u,\bar v)}.
$$
As standard error is defined as
$$
\delta=\frac \sigma {\sqrt{N}},
$$

$$
\sigma^2 = N\cdot\delta^2.
$$

Hence
$$
N\cdot \delta_x^2 \approx N\cdot(\delta^2_uf_u^2+\delta^2_vf_v^2).
$$

$$
\delta^2_x \approx \delta^2_uf_u^2+\delta^2_vf_v^2.
$$

where $f_u,f_v$ are both evaluated at $(\bar u, \bar v).$

## 3

### a)

$$
\sigma_x\approx \sqrt{\frac{\sigma_u^2+\sigma_v^2}{4(\bar u+\bar v)^4}}=\frac{\sqrt{{\sigma_u^2+\sigma_v^2}}}{2(\bar u+\bar v)^2}
$$

### b)

$$
\sigma_x\approx \sqrt{\frac{\sigma_u^2+\sigma_v^2}{4(\bar u-\bar v)^4}}=\frac{\sqrt{{\sigma_u^2+\sigma_v^2}}}{2(\bar u-\bar v)^2}
$$

### c)

$$
\sigma_x\approx \sqrt{\frac{4\sigma_u^2}{\bar u^6}}=\frac{2\sigma_u}{|\bar{u}^3|}
$$

### d)

$$
\sigma_x\approx \sqrt{\sigma_u^2\bar v^4+4\sigma_v^2\bar u^2\bar v^2}
$$

### e)

$$
\sigma_x\approx 2\sqrt{\sigma_u^2\bar u^2+\sigma_v^2\bar v^2}
$$

### f)

$$
\sigma_x\approx \frac{|ab\cos(b\bar u/\bar v)|}{\bar v^2}\sqrt{{\sigma_u^2\bar v^2+\sigma_v^2}\bar u^2}
$$

### g)

$$
\sigma_x\approx \sqrt{\sigma_u^2a^2/\bar u^2}=\sigma_u|{a/\bar u}|
$$

### h)

$$
\sigma_x\approx e^{b\bar u+c\bar v}\sqrt{\sigma_u^2b^2+\sigma_v^2c^2}
$$

### i)

$$
\sigma_x=0
$$

## 4

In radians, $\theta_1=0.1224\pi\pm0.001111\pi,\theta_2=0.08028\pi\pm0.001111\pi.$
$$
\begin{aligned}
\bar n_2(\bar\theta_1,\bar\theta_2,n_1)&=\frac {\sin\bar\theta_1}{\sin\bar\theta_2}n_1\\
&=\frac {\sin(0.1224\pi)}{\sin(0.08028\pi)}\cdot1.0000\\
&=1.503.

\end{aligned}
$$

$$
\begin{aligned}
\delta_{n_2}&=\csc\bar\theta_2\sqrt{\delta^2_{\theta_1}\cos^2\bar\theta_1+\delta^2_{\theta_2}\sin^2\bar\theta_1\cot^2\bar\theta_2}\\&=
0.024.
\end{aligned}
$$

Therefore,
$$
n_2=1.503\pm0.024.
$$

## 5

$$
\begin{aligned}
\bar g&=\frac{4\pi^2t}{\bar T^2}\\
&=9.801\  \text{m}\cdot\text{s}^{-2}.

\end{aligned}
$$

$$
\begin{aligned}
\delta_g&=4\pi^2\sqrt{4\delta_T^2\bar t^2\bar T^{-6}+\delta_t^2\bar T^{-4}}\\
&=0.028\  \text{m}\cdot\text{s}^{-2}.
\end{aligned}
$$

Therefore,
$$
g=9.801\pm 0.028\   \text{m}\cdot\text{s}^{-2}.
$$

## 6

The mean speed of the disk is
$$
\omega_i(\theta_i,\theta_{i-1},t_i,t_{i-1})=\frac{\Delta\theta_i}{\Delta t_i}=1000\  °\text{s}^{-1}=17.45\ \text{s}^{-1}=20\ \text{s}^{-1}.
$$
If $\delta_\theta=0.09°=0.00157\ \text{rad}, \delta _t=10\  \mu\text s,$ then
$$
\begin{aligned}
\delta_{\omega_i}&=\sqrt{(\delta_{\theta_i}^2+\delta_{\theta_{i-1}}^2)(\Delta t_i)^{-2}+(\delta_{t_i}^2+\delta_{t_{i-1}}^2)(\Delta\theta_i)^2(\Delta t_i)^{-4}}\\&=
\sqrt 2\sqrt{\delta_\theta^2(\Delta t_i)^{-2}+\delta_t^2(\Delta\theta_i)^2(\Delta t_i)^{-4}}\\&=
2.235\ \text s^{-1}\\&=
2\ \text s^{-1},
\end{aligned}
$$
and
$$
\begin{aligned}
\delta_v&=\sqrt{\delta_R^2\omega^2+\delta_\omega^2R^2}\\&=
\sqrt{(.1\text{mm})^2(17.45\ \text{s}^{-1})^2+(2.235\ \text s^{-1})^2(20\ \text{mm})^2}
\\&=
0.0447\ \text m\cdot \text s^{-1}\\&=
0.04\ \text m\cdot \text s^{-1}.
\end{aligned}
$$



