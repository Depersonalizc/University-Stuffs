# EIE2050 Assignment 1

## 1

### (a)

$$
(100001)_2=2^5+2^0=(33)_{10}
$$

### (b)

$$
(100111)_2=2^5+2^2+2^1+2^0=(39)_{10}
$$

### (c)

$$
(101010)_2=2^5+2^3+2^1=(42)_{10}
$$

### (d)

$$
(111001)_2=2^5+2^4+2^3+2^1=(57)_{10}
$$

### (e)

$$
(1100000)_2=2^6+2^5=(96)_{10}
$$

### (f)

$$
(11111101)_2=(100000000-1-10)_2=2^8-1-2^1=(253)_{10}
$$

### (g)

$$
(11110010)_2=(100000000-1-1101)_2=2^8-1-2^3-2^2-2^0=(242)_{10}
$$

### (h)

$$
(11111111)_2=(100000000-1)_2=2^8-1=(255)_{10}
$$

## 2

### (a) 

$$
\begin{array}{c|cc}
&.76\\
1&.52\\
1&.04\\
0&.08\\
0&.16\\
0&.32\\\vdots&\vdots
\end{array}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (0.76)_{10}\approx (0.11000)_2
$$

### (b) 

$$
\begin{array}{c|cc}
&.456\\
0&.912\\
1&.824\\
1&.648\\
1&.296\\
0&.592\\\vdots&\vdots
\end{array}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (0.456)_{10}\approx(0.01110)_2
$$



### (c) 

$$
\begin{array}{c|cc}
&.8732\\
1&.7464\\
1&.4928\\
0&.9856\\
1&.9712\\
1&.9424\\\vdots&\vdots
\end{array}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (0.8732)_{10}\approx(0.11011)_2
$$

## 3

*(in binary arithmetic)*

### (a)

$$
\begin{aligned}
11\div 11&=\textcolor{red}{1}\cdots\textcolor{blue}{0}\\
(\textcolor{blue}{0}\times 10+0)\div11&=\textcolor{red}{0}
\end{aligned}\implies
110\div11=\boxed{10}
$$

### (b)

$$
\begin{aligned}
10\div 10&=\textcolor{red}{1}\cdots\textcolor{blue}{0}\\
(\textcolor{blue}{0}\times10+1)\div 10 &=\textcolor{red}{0}\cdots\textcolor{blue}{1}\\
(\textcolor{blue}{1}\times10+0)\div 10 &=\textcolor{red}{1}
\end{aligned}
\implies1010\div10=\boxed{101}
$$

### (c)

$$
\begin{aligned}
111\div 101&=\textcolor{red}{1}\cdots \textcolor{blue}{10}\\
(\textcolor{blue}{10}\times10+1)\div101&=\textcolor{red}{1}
\end{aligned}
\implies1111\div101=\boxed{11}
$$

## 4

### (a)

With $8$-bits, the additive inverse of $(10011001)_2$ is
$$
(100000000-10011001)_2=(01100111)_2=(103)_{10}
$$
Therefore the decimal value of $10011001$ (in 2's complement) is $\boxed{-1 03.}$

### (b)

Sign bit is $0. $ Positive number has itself as 2's complement:
$$
(01110100)_2=(116)_{10}
$$
Therefore the decimal value of $01110100$ (in 2's complement) is $\boxed{116.}$

### (c)

The additive inverse of $(10111111)_2$ in 8-bits is
$$
(100000000-10111111)_2=(01000001)_2=(65)_{10}
$$
Therefore the decimal value of $10111111$ (in 2's complement) is $\boxed{-65.}$

## 5

*(in binary arithmetic)*

### (a)

$$
\begin{aligned}
&\ \ \ \ \ (-1)^1\times 1.01001001110001\times 10^{(10000001-01111111)}\\&=\boxed{-1.01001001110001\times 10^2}
\end{aligned}
$$

### (b)

$$
\begin{aligned}
&\ \ \ \ \ (-1)^0\times 1.100001111101001\times 10^{(11001100-01111111)}\\&=\boxed{1.100001111101001\times10^{77}}
\end{aligned}
$$

## 6

*(in binary arithmetic)*

### (a)

$$
\begin{array}{c}
\begin{aligned}
\text{Bor}\ \ 0&\ 0000 \ 000   \\
   &\ 0011 \ 0011  \\
 −\ \ \ \  &\ 0001 \ 0000  \\
 \hline
   &\boxed{0010 \ 0011 } 
   \end{aligned}
 \end{array}
$$

### (b)

$$
\begin{array}{c}
\begin{aligned}
\text{Bor}\ \  1&\ 1111 \ 000 \\
   &\ 0110 \ 0101  \\
 − \ \ \ \ &\ 1110 \ 1000  \\
 \hline
   &\boxed{0111 \ 1101  }
   \end{aligned}
 \end{array}
$$

## 7

*(in hexadecimal arithmetic)*

### (a)

$$
\begin{array}{c}
\begin{aligned}
\text{Bor}\ \  &\ 1\\
   &\ 60  \\
 − \ \ &\ 39  \\
 \hline
     &\boxed{27}
   \end{aligned}
 \end{array}
$$

### (b)

$$
\begin{array}{c}
\begin{aligned}
\text{Bor}\ \  &\  1\\
   &\text{A}5  \\
 − \ \ &\ 98  \\
 \hline
     &\ \  \boxed{\text{D}}
   \end{aligned}
 \end{array}
$$

### (c)

$$
\begin{array}{c}
\begin{aligned}
\text{Bor}\ \  &  1\\
   &\text{F}1  \\
 − \ \ & \text{A}6  \\
 \hline
     & \boxed{4\text{B}}
   \end{aligned}
 \end{array}
$$

### (d)

$$
\begin{array}{c}
\begin{aligned}
\text{Bor}\ \  &\  0\\
   &\text{AC}  \\
 − \ \ &\ 1\ 0  \\
 \hline
     &\boxed{9\text{C}}
   \end{aligned}
 \end{array}
$$

## 8

### (a)

$$
4=(0100)_{\text{BCD}};3=(0011)_{\text{BCD}}\\ 4+3=(0111)_{\text{BCD}}
$$

### (b)

$$
5=(0101)_{\text{BCD}};2=(0010)_{\text{BCD}}\\ 5+2=(0111)_{\text{BCD}}
$$

### (c)

$$
6=(0000\ 0110)_{\text{BCD}};4=(0000\ 0100)_{\text{BCD}}\\ 6+4=(0000\ 1010)_2\overset{+0110}=(0001\ 0000)_{\text{BCD}}
$$

### (d)

$$
17=(0001\ 0111)_{\text{BCD}};12=(0001\ 0010)_{\text{BCD}}\\ 17+12=(0010\ 1001)_{\text{BCD}}
$$

### (e)

$$
28=(0010\ 1000)_{\text{BCD}};23=(0010\ 0011)_{\text{BCD}}\\28+23=(0100\ 1011)_2\overset{+0110}{=}(0101\ 0001)_{\text{BCD}}
$$

### (f)

$$
65=(0110\ 0101)_{\text{BCD}};58=(0101\ 1000)_{\text{BCD}}\\ 65+58=(1011\ 1101)_2\overset{+0110\ 0110}{=}(0001\ 0010\ 0011)_{\text{BCD}}
$$

### (g)

$$
113=(0001\ 0001\ 0011)_{\text{BCD}};101=(0001\ 0000\ 0001)_{\text{BCD}} \\ 113+101=(0010\ 0001\ 0100)_{\text{BCD}}
$$

### (h)

$$
295=(0010\ 1001\ 0101)_{\text{BCD}};157=(0001\ 0101\ 0111)_{\text{BCD}} \\ 295+157=(0011\ 1110\ 1100)_2\overset{+0110\ 0110}{=}(0100\ 0101\ 0010)_{\text{BCD}}
$$

## 9

$$
\bold{Hello.\ How\ are\ you?}
$$

## 10

$\bold{(a)}$ and $\bold{(c)}$ are in error because parity is even $(6\text{ and }8).$