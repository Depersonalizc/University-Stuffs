# Sparse Reconstruction in Image Inpainting

The $\ell_1-$regularized image reconstruction problem is an alternative method to solve the image inpainting problem. The formulation is:
$$
\min_{x\in\R^{mn}}\  ||\Psi x||_1\qquad\text{s.t.}\qquad ||Ax-b||_\infty\le\delta,\tag3
$$
where $\Psi_{mn\times mn}$ transfers the image $x$ to the frequency domain; $A_{s\times mn}$ and $b\in\R^s$ are as in the Total Variation Minimization Problem formulation, and $\delta\gt\mathbb0$ is the error threshold between the undamaged pixels $b$ and the reconstructed version $Ax.$

## LP Formulation

We first reformulate $(3)$ as a linear program by noting that
$$
\begin{aligned}
&\quad\ \min_{x\in\R^{mn}}\  ||\Psi x||_1\qquad\text{s.t.}\qquad ||Ax-b||_\infty\le\delta\\

&=\min_{x\in\R^{mn}}\  \mathbf1^\top |\Psi x|\qquad\text{s.t.}\qquad \left|A x-b\right|\le\delta\mathbf1\\

&=\min_{x,t\in\R^{mn}}\  \mathbf1^\top t\qquad\text{s.t.}\qquad t\ge\pm\Psi x, \quad \pm\left(A x-b\right)\le\delta\mathbf1.

\end{aligned}\tag4
$$
Here, $\mathbf1$ denotes the all-one vectors of appropriate sizes and $|v|$ is interpreted element-wise on vector $v.$

## Dual Problem

Next, we derive the associated dual of $(4).$ Rewrite $(4)$ in matrix form
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
\end{array}\right],
\end{aligned}
$$
where $I_r$ denotes the $r\times r$ identity matrix and $0_{p\times q}$ is the $p\times q$ all-zero matrix.

The dual is then given by
$$
\begin{aligned}
&\quad\ \max_{u,v\in\R^{mn}\\y,z\in\R^{s}} \left[0^\top|\ 0^\top|-\delta\mathbf1^\top+ b^\top\ |-\delta\mathbf1^\top- b^\top\right]\left[\begin{array}{c}
u\\v\\y\\z
\end{array}\right]
\\
&\qquad\ \text{s.t.}\qquad
\left[\begin{array}{cc}
\Psi^\top&-\Psi^\top&A^\top&-A^\top\\
I_{mn}&I_{mn}&0_{mn\times s}&0_{mn\times s}\\
\end{array}\right]
\left[\begin{array}{c}
u\\v\\y\\z
\end{array}\right]
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