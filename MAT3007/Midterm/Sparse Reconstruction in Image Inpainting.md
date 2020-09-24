

# $\ell_1$ Image Inpainting

The $\ell_1$-regularized image reconstruction is an alternative method to solve the image inpainting problem:
$$
\min_{x\in\R^{mn}}\  ||\Psi x||_1\qquad\text{s.t.}\qquad ||Ax-b||_\infty\le\delta,\tag1
$$
where $\Psi_{mn\times mn}$ transfers the image $x$ to the frequency domain via Discrete Cosine Transformation; $A_{s\times mn}$ and $b\in\R^s$ are as in Total Variation Minimization, and $\delta\gt\mathbb0$ is the error threshold for reconstruction $Ax$ at undamaged pixels $b.$ 

We derived the linear programming formulation of $(1)$ as well as its dual problem. We implemented and tested the the program on sample grey-scale images. We examined the reconstruction quality and speed of the model under different sets of parameters.

## LP formulation

We first reformulate $(1)$ as a linear program by noting that
$$
\begin{aligned}
&\quad\ \min_{x\in\R^{mn}}\  ||\Psi x||_1\qquad\text{s.t.}\qquad ||Ax-b||_\infty\le\delta\\

&=\min_{x\in\R^{mn}}\  \mathbf1^\top |\Psi x|\qquad\text{s.t.}\qquad \left|A x-b\right|\le\delta\mathbf1\\

&=\min_{x,t\in\R^{mn}}\  \mathbf1^\top t\qquad\text{s.t.}\qquad t\ge\pm\Psi x, \quad \pm\left(A x-b\right)\le\delta\mathbf1.

\end{aligned}\tag2
$$
Here, $\mathbf1$ denotes the all-one vectors of appropriate sizes and $|v|$ is interpreted element-wise on vector $v.$

## Dual problem

Next, we derive the associated dual of $(1).$ Rewrite $(2)$
$$
\begin{aligned}
&\quad\ \min_{x,t\in\R^{mn}}\  \mathbf1^\top t\qquad\text{s.t.}\qquad t\ge\pm\Psi x, \quad \pm\left(A x-b\right)\le\delta\mathbf1
\\&=\min_{x,t\in\R^{mn}}\mathbf1^\top t\qquad\text{s.t.}\qquad 
\pm\Psi x+t\ge0,\quad\pm A x\ge-\delta\mathbf1\pm b\\
&=\min_{x,t\in\R^{mn}} \left[0^\top|\ \mathbf1^\top\right]\left[\begin{array}{c}
x\\t
\end{array}\right]\qquad\text{s.t.}\qquad
\left[\begin{array}{cc}
\Psi&I_{mn}\\
-\Psi&I_{mn}\\
A& 0_{s\times mn}\\
-A&0_{s\times mn}
\end{array}\right]
\left[\begin{array}{c}
x\\t
\end{array}\right]\ge
\left[\begin{array}{c}
0\\0\\-\delta\mathbf1+ b
\\-\delta\mathbf1- b
\end{array}\right].
\end{aligned}
$$
where $I_r$ denotes the $r\times r$ identity matrix and $0_{p\times q}$ is the $p\times q$ all-zero matrix.

The dual is then given by
$$
\begin{aligned}
&\quad\ \max_{u,v\in\R^{mn}\\y,z\in\R^{s}} \left[0^\top|\ 0^\top|-\delta\mathbf1^\top+ b^\top\ |-\delta\mathbf1^\top- b^\top\right]\left[\begin{array}{c}
u & v & y & z
\end{array}\right]^\top
\\
&\qquad\ \text{s.t.}\qquad
\left[\begin{array}{cc}
\Psi^\top&-\Psi^\top&A^\top&-A^\top\\
I_{mn}&I_{mn}&0_{mn\times s}&0_{mn\times s}\\
\end{array}\right]
\left[\begin{array}{c}
u&v&y&z
\end{array}\right]^\top
=
\left[\begin{array}{c}
0\\\mathbf1
\end{array}\right],\quad u,v,y,z\ge0\\\\&=
\max_{u,v\in\R^{mn}\\y,z\in\R^{s}}\ b^\top(y-z)-\delta\mathbf1^\top(y+z)\\
&\qquad\text{s.t.}\qquad \Psi^\top(u-v)+A^\top(y-z)=0,\quad u+v= \mathbf1,\quad u,v,y,z\ge0\\\\&=
\max_{t\in\R^{mn}\\r,w\in\R^{s}}\ b^\top w-\delta\mathbf1^\top(w+r)\\
&\qquad\ \text{s.t.}\qquad \Psi^\top t+A^\top w=0,\quad r\ge0,\quad -1\le t\le1\\\\&=
\max_{t\in\R^{mn}\\w\in\R^{s}}\ \left(b^\top -\delta\mathbf1^\top\right) w\\
&\qquad\text{s.t.}\qquad \Psi^\top t+A^\top w=0,\quad -1\le t\le1.
\end{aligned}
$$

## Implementation

We implemented the $\ell_1$ inpainting algorithm with `linprog` function in MATLAB R2020a. The `interior-point` method was chosen for speed concerns.

```matlab
...

% linprog
% min(c'x) s.t. Mx <= d
del     = delta*ones(s, 1);
I       = speye(m*n);
ze      = sparse(s, m*n);
c       = [zeros(m*n, 1); ones(m*n, 1)];
M       = [-Psi -I; Psi -I; -A ze; A ze];
d       = [zeros(2*m*n, 1); del-b; del+b];
options = optimoptions('linprog', 'Algorithm', 'interior-point', ...
                       'ConstraintTolerance', 1e-3, 'Display', 'iter');
x       = linprog(c, M, d, [], [], [], [], options);

...
```

(For complete source code, please refer to the attached `.m` files.)

## Results

We tested the model on sample grey-scale images of $256\times256$ and $512\times512$ pixels. The results were obtained with a 4.0 GHz Quad-Core Intel Core i7-4790K processor and 32 GB 2133 MHz memory. The quality of the reconstructed images was assessed via the PSNR value:
$$
\text{PSNR}:=10\cdot\log_{10}\frac{mn}{||x-u^*||^2},
$$
where $x$ is the reconstructed image and $u^*=\text{vec}(U^*)$ is the ground truth.

### Overall performance

The $\ell_1$ block model reconstructs $256\times256$ images contaminated by $50\%$ random noise with $\text{PSNR}\approx 20.$ The average runtime is around $40\text{ sec}$ with a maximum of $20$ iterations. We notice that the algorithm takes significantly longer at each iteration, and more iterations to converge, if the block size $\text{bsz}$ is set at $16$ (the value recommended by the lecturer for low-res images). For instance, image $(\mathbf a)$ takes nearly $8\text{ min}$ to reconstruct at $\text{bsz}=16,$ along with a $2.3$ increase in PSNR, and images $(\mathbf b)$ and $(\mathbf c)$ fail to converge within a $200$-iteration limit. After examining matrix $\Psi,$ we find that it is because the number of non-zero elements $\Psi$ increase proportionally to $\text{bsz}^2,$ resulting in significantly more computations.

|                   | Ground Truth                                                 | Ground Truth + $50\%$ Noise                                  | Reconstructed Images                                         | PSNR     | Runtime (Iterations)        |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | --------------------------- |
| $(\mathbf a)$     | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\c.PNG" alt="c" style="zoom: 25%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\masked\2 (2).png" alt="2 (2)" style="zoom: 25%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\buildings_bsz_8_21.6.png" alt="buildings_bsz_8_21.6" style="zoom: 25%;" /> | $21.6$   | $38.4\text{ sec}\\(17)$     |
| $(\text{bsz}=16)$ |                                                              |                                                              | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\256\256_house+rand_50_del0.06_474s.png" alt="256_house+rand_50_del0.06_474s" style="zoom: 25%;" /> | $23.9$   | $7.9\text{ min}$            |
| $(\mathbf b)$     | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\B.PNG" alt="B" style="zoom: 25%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\masked\2.png" alt="2" style="zoom: 25%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\256\256_hand+random_50_del0.06_s.png" alt="256_hand+random_50_del0.06_48s" style="zoom: 25%;" /> | $22.3$   | $44.2\text{    sec}\\(20)$  |
| $(\mathbf c)$     | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\A.PNG" alt="A" style="zoom: 25%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\masked\casino.png" alt="casino" style="zoom: 25%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\256\256_casino+rand_50_del0.06_49s.png" alt="256_casino+rand_50_del0.06_49s" style="zoom: 25%;" /> | $$20.4$$ | $44.5\text{    sec}\\ (19)$ |

*Table 1.* $\ell_1$ inpainting results of $256\times256$ images contaminated by $50\%$ random noise, $\delta=6\times10^{-2},$ $\text{bsz}=8,$ constraint tolerance $= 10^{-3},$ optimality tolerance $= 10^{-6}$

<img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\Psi\bsz.png" alt="bsz" style="zoom: 70%;" />

*Figure 1.* Sparsity patterns of $\Psi$ with $\text{bsz}=32$ (Left) and $\text{bsz}=64$ (Right), $m=n=256$

In the task of reconstructing $512\times512$ image polluted by $50\%$ random noise, the algorithm achieves roughly the same $\text{PSNR}$ compared to low-res images within similar number of iterations, while the runtime becomes roughly four times as long. We also observed that as the noise intensity increased, the image quality deteriorates drastically although the algorithm seems to converge faster.

| Noise Percentage         | Ground Truth                                                 | $30\%$                                                       | $50\%$                                                       | $70\%$                                                       |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Ground Truth + Noise** | **![512_512_lena](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_images\512_512_lena.png)** | **![lena_rand30](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\masked\lena_rand30.png)** | **![lena_rand50](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\masked\lena_rand50.png)** | **![lena_rand70](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\masked\lena_rand70.png)** |
| **Reconstructed Images** | **-**                                                        | **![512_lena+rand_30_del0.06_265s_25.5](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\noise_cmp\512_lena+rand_30_del0.06_265s_25.5.png)** | **![512_lena+rand_50_del0.06_186s_22.6](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\noise_cmp\512_lena+rand_50_del0.06_186s_22.6.png)** | **![512_lena+rand_70_del0.06_170s_17](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\noise_cmp\512_lena+rand_70_del0.06_170s_17.png)** |
| **PSNR**                 | **-**                                                        | **$25.5$**                                                   | **$22.6$**                                                   | **$17.0$**                                                   |
| **Runtime (Iterations)** | -                                                            | $4.4\text{ min}\ (26)$                                       | $3.1\text{ min}\ (21)$                                       | $2.8\text{ min}\ (21)$                                       |

*Table 2.* $\ell_1$ inpainting results of $512\times512$ images contaminated by $30\%,50\%,$ and $70\%$ random noises, $\delta=6\times10^{-2}, \text{bsz}=8,$ constraint tolerance $= 10^{-3},$ optimality tolerance $= 10^{-6}$

In reconstructing images with non-random damages, the algorithm performs roughly the same as in the $50\%$ random noise case with mesh, handwriting, and mild scratches. However, when inpainting the hard-scratched image, the algorithm performs poorly with $\text{PSNR} = 13.9,$ a score even lower than with the $70\%$ random noise, which covers more area of the image than the scratches. From this, we conclude that the algorithm is sensitive to the distribution of the damage. In particular, the algorithm performs better if the damage distributes more evenly, i.e., there is no large "chunks" of damage area on the image (as in hard scratches).

| Damage Types             | Ground Truth                                                 | Mesh                                                         | Handwriting                                                  | Mild Scratches                                               | Hard Scratches                                               |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Damaged Images**       | ![512_512_lena](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_images\512_512_lena.png) | ![mesh](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\mesh.png) | ![handwritting](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\handwritting.png) | ![scratch1](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\scratch1.png) | ![scratch](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\scratch.png) |
| **Reconstructed Images** | -                                                            | ![512_lena+mesh_del0.06_253s_21.6](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\mask_cmp\512_lena+mesh_del0.06_253s_21.6.png) | ![512_lena+writing_50_del0.06_253s_21.5](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\mask_cmp\512_lena+writing_50_del0.06_253s_21.5.png) | ![scratch1_FIX](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\scratch1_FIX.png) | ![scratch_FIX](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\scratch_FIX.png) |
| **PSNR**                 | -                                                            | $21.6 $                                                      | $21.5$                                                       | $25.3$                                                       | $13.9$                                                       |
| **Runtime (Iterations)** | -                                                            | $4.2\text{ min}\ (26)$                                       | $4.2\text{ min}\ (21)$                                       | $4.4\text{ min}\ (24)$                                       | $3.7\text{ min}\ (22)$                                       |

*Table 3.* $\ell_1$ inpainting results of $512\times512$ images with non-random damage, $\delta=6\times10^{-2},$ $\text{bsz}=8,$ constraint tolerance $= 10^{-3},$ optimality tolerance $= 10^{-6}$

### Effects of adjusting termination tolerances

We investigate the effects of adjusting optimality tolerance on the reconstruction quality by inpainting the $50\%$ random noise contaminated image. We observe almost no changes either in image quality or runtime, when optimality tolerance ranges from $10^{-7}$ to $10^{-5}.$ Curiously, when optimality tolerance is larger than $10^{-4},$ the algorithm gets significantly slower and fails to converge within $200$ iterations.

| Optimality Tolerance     | Ground Truth                                                 | $10^{-3,-4}$ | $10^{-5}$                                                    | $10^{-6}$                                                    | $10^{-7}$                                                    |
| ------------------------ | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Reconstructed Images** | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_images\512_512_lena.png" alt="512_512_lena"  /> | -            | ![optol_e-5](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\optol_cmp\optol_e-5.png) | ![512_lena+rand_50_del0.06_186s_22.6](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\delta_cmp\512_lena+rand_50_del0.06_186s_22.6.png) | ![optol_e-7](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\optol_cmp\optol_e-7.png) |
| **PSNR**                 | -                                                            | -            | $22.6$                                                       | $22.6$                                                       | $22.6$                                                       |
| **Runtime (Iterations)** | -                                                            | $(>200)$     | $4.4\text{ min }(29)$                                        | $3.1\text{ min}\ (21)$                                       | $3.5\text{ min}\ (23)$                                       |

*Table 4.* Effects of adjusted optimality tolerance on $\ell_1$ inpainting results of $512\times512$ images contaminated by $50\%$ random noise, $\delta=6\times10^{-2}, \text{bsz}=8,$ constraint tolerance $= 10^{-3}$

Our results also suggest the constraint tolerance has no significant impacts on reconstruction quality or speed. However when constraint tolerance is below $10^{-6},$ the algorithm is significantly slower and struggles to converge within the $200$-iteration limit.

| Constraint Tolerance     | Ground Truth                                                 | $10^{-3}$                                                    | $10^{-4}$                                                    | $10^{-5}$                                                    | $10^{-6,-7}$ |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **Reconstructed Images** | ![512_512_lena](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_images\512_512_lena.png) | ![512_lena+rand_50_del0.06_186s_22.6](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\delta_cmp\512_lena+rand_50_del0.06_186s_22.6.png) | ![contol_e-4](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\contol_cmp\contol_e-4.png) | ![contol_e-5](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\contol_cmp\contol_e-5.png) | -            |
| **PSNR**                 | -                                                            | $22.6$                                                       | $22.6$                                                       | $22.6$                                                       | -            |
| **Runtime (Iterations)** | -                                                            | $3.1\text{ min}\ (21)$                                       | $3.5\text{ min}\ (22)$                                       | $3.4\text{ min}\ (22)$                                       | $(\gt200)$   |

*Table 5.* Effects of adjusted constraint tolerance on $\ell_1$ inpainting results of $512\times512$ images contaminated by $50\%$ random noise, $\delta=6\times10^{-2}, \text{bsz}=8,$ optimality tolerance $= 10^{-6}$

### Effects of adjusting $\delta$ and comparison to the TV model ###

We investigate the effects of altering the error threshold $\delta$ on the reconstruction quality by inpainting the $50\%$ random noise contaminated image. We observe that the image quality slightly decreases as the threshold grows from $0.006$ to $0.06,$ but is still within the acceptable range. Nevertheless, when $\delta$ reaches $0.3,$ the reconstructed image becomes visibly darker and coarser, and at $\delta=0.6,$ the image is almost unidentifiable.

Compared with the Total Variation model in part 1 of the project, the $\ell_1$ block model is significantly inferior in both reconstruction quality and speed. The excessive runtime of the $\ell_1$ model is partly due to a denser coefficient matrix compared with TV model. In this example, the density of the coefficient matrix in the $\ell_1$ model is $8.33\times10^{-5}$ while in the TV model, the number is $3.82\times 10^{-6},$ almost two magnitudes lower.

| $\delta$                 |                       **Ground Truth**                       |                     TV (Interior Point)                      |                           $0.006$                            |                            $0.06$                            |                            $0.3$                             |                            $0.6$                             |
| :----------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Reconstructed Images** | ![512_512_lena](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_images\512_512_lena.png) | ![lena-random50-interior-point-34.5214-19.0156](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\lena-random50-interior-point-34.5214-19.0156.png) | ![512_lena+rand_50_del0.006_276s_26.3](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\delta_cmp\512_lena+rand_50_del0.006_276s_26.3.png) | ![512_lena+rand_50_del0.06_186s_22.6](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\delta_cmp\512_lena+rand_50_del0.06_186s_22.6.png) | ![512_lena+rand_50_del0.3_196s_11.3](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\delta_cmp\512_lena+rand_50_del0.3_196s_11.3.png) | ![512_lena+rand_50_del0.6_194s_6.5](C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\delta_cmp\512_lena+rand_50_del0.6_194s_6.5.png) |
| **PSNR**                 |                              -                               |                            $34.5$                            |                            $26.3$                            |                            $22.6$                            |                            $11.3$                            |                            $6.5$                             |
| **Runtime (Iterations)** |                              -                               |                   $11.0\text{ sec}\ (11)$                    |                    $4.6\text{ min}\ (30)$                    |                    $3.1\text{ min}\ (21)$                    |                    $3.3\text{ min}\ (22)$                    |                    $3.2\text{ min}\ (21)$                    |

*Table 6.* Effects of adjusted $\delta$ on $\ell_1$ inpainting results of $512\times512$ images contaminated by $50\%$ random noise, $\text{bsz}=8,$ constraint tolerance $= 10^{-3},$ optimality tolerance $= 10^{-6},$ TV model result included for comparison

### Denoising

We finally test the $\ell_1$ block model under the denoising setting: $A=I_{mn},b=u+\sigma\text{randn}(mn,1)$ where $A,b,u$ are as in the TV minimization model, and $\text{randn}$ adds scaled Gaussian white noise to the stacked image $u$. The following results are obtained with parameters set at $\sigma=0.1,\delta=0.9\sigma,\text{bsz}=8.$

|                       **Ground Truth**                       |                     GT + Gaussian Noise                      |                        Denoised Image                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_images\512_512_lena.png" alt="512_512_lena" style="zoom:67%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\denoise\noised.PNG" alt="noised" style="zoom:67%;" /> | <img src="C:\Users\Jamie\Documents\University-Stuffs\MAT3007\Midterm\test_results\denoise\denoised.PNG" alt="denoised" style="zoom:67%;" /> |
|                           **PSNR**                           |                            $5.7$                             |                            $22.8$                            |

*Table 7.* $\ell_1$ denoising result of $512\times512$ images contaminated by Gaussian white noise, $\sigma=0.1,\delta=0.9\sigma,\text{bsz}=8.$

## Conclusion

The overall performance of the $\ell_1$ image inpainting model were unsatisfactory in terms of reconstruction quality and running speed. On all sample images we tested, the $\text{PSNR}$ of the model's reconstruction results fell well short of $30,$ a benchmark easily obtained by the TV model; the speed of the $\ell_1$ model was also significantly slower than the TV model. With $\text{bsz}=8$ the typical runtime of reconstructing a $512\times512$ image was $3\text{ min}$ - $5 \text{ min}$ in $20$ -  $30$ iterations, while the TV model could achieve better results in around $10$ seconds, $10$ iterations. The reason for the slow running speed might be due to a relatively dense $\Psi$ matrix, which led to a large amount of computation. Better reconstruction results might be obtained by choosing larger $\text{bsz}$ such as $16$ or $32,$ however the density of $\Psi$ would also grow quadratically to $\text{bsz},$ rendering the task almost impossible on a regular PC. 

We discovered that the model produced better results when reconstructing images with evenly-spread damage - which might be explained by the damage's correspondence to higher frequencies after transformed by $\Psi.$ We found that neither the optimality tolerance nor the constraint tolerance had a significant impact on the quality or runtime of the reconstruction. Nevertheless, the program might experience instability and some indefinite behavior with optimality tolerance $\gt10^{-4}$ or constraint tolerance $\lt10^{-6}.$ We also found that smaller $\delta$ produced better reconstruction results, while sacrificing some speed when $\delta<0.01.$ The program might experience instability when $\delta$ was below $0.006.$ Finally, we found that the $\ell_1$ model might be suitable for denoising tasks (in which the positions of noise signals are unknown).

