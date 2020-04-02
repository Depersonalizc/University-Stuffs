using Plots
n = BigInt(100); p = .1; λ=10; x = BigInt.(0:n);
bi(x) = binomial(n, x) * p^x * (1 - p)^(n - x)
poi(x) = λ^x / (ℯ^λ * factorial(x))
x1 = plot(bi.(x), line=:stem, label="Binomial(100, 0.1)")
x2 = plot(poi.(x), line=:stem, label="Poisson(10)", color=:red)
plot(x1,x2)