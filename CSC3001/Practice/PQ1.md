# Exercise 1

## 1

### (a)

F

### (b)

T

### (c)

F

### (d)

F

### (e)

T

### (f)

T

### (g)

F

## 2

### (a)

$\neg A.$

### (b)

$A\and B.$

### (c)

$A\rightarrow \neg B$

### (d)

$A\or (\neg A \rightarrow B).$

### (e)

$(A\and B)\or (\neg A \and \neg B).$

## 3

### (a)

$p\oplus q=(p\and \neg q)\or (\neg p \and q).$

### (b)

$p\rightarrow q=\neg(p\and \neg q)=\neg p \or q$

### (c)

$p\odot q =(\neg p \or q)\and (p\or \neg q)=(p\and q)\or(\neg p\and \neg q).$

### (d)

$\neg(p\rightarrow q)=\neg\neg(p\and \neg q)=p\and \neg q.$

## 4

### (a)

$$
\begin{aligned}f(p,q,r)&=(p\and q\and r)\or ((\neg p\and \neg q \and r)\or (\neg p\and \neg q\and \neg r))\\&=(p\and q\and r)\or ((\neg p\and \neg q) \and (r\or \neg r))\\&=(p\and q\and r)\or (\neg p\and \neg q) .\end{aligned}
$$

### (b)

$$
\begin{aligned}f(p,q,r)&=\neg((p\and q\and \neg r)\or (\neg p\and q \and \neg r)\or(\neg p\and \neg q\and \neg r))\\&=(\neg p\or \neg q\or r)\and ( p\or \neg q\or r)\and (p\or q\or r)\\&=(\neg q\and(p\or q))\or r\\&=(p\and \neg q)\or r.\end{aligned}
$$

## 5

### (a)

Not equivalent. 

If $(p,q,r)=(F,F,T),$ then
$$
\begin{aligned}
pq+r&=FF+T=F+T=T\\
&\neq F=FT=F(F+T)=p(q+r).\end{aligned}
$$

### (b)

Equivalent.
$$
\begin{aligned}pq\bar r+p\bar q+r&=p(q\bar r+\bar q)+r\\&=\bar rp(q\bar r+\bar q)+r\\&=p(q\bar r+\bar q\bar r)+r\\&=p\bar r+r=p+r.\end{aligned}
$$

### (c)

Equivalent.
$$
\begin{aligned}
\neg{(p+q+r)}&=\neg((p+q)+r)\\&=\neg(p+q)\neg r\\&=(\neg p\neg q)\neg r\\&=\neg p\neg q\neg r.
\end{aligned}
$$

### (d)

Equivalent.
$$
p(p+q)=(p+p)(p+q)=p+(pq).
$$

### (e)

Not equivalent.

If $(p,q,r)=(T,F,T),$ then
$$
\begin{aligned}(pq)+(qr)&=(p+r)q=(T+T)F=F\\&\neq T=F+(TT)=q+(pr).\end{aligned}
$$

