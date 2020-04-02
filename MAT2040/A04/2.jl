using Plots
n = 10; N = 1e6; p = .5
rslt = Dict([(x,0) for x=0:n])
for _ = 1:N
    c = count(x->(rand()<p), 1:n)
    rslt[c] += 1
end
f(x) = rslt[x] / N
bi(x) = binomial(n, x) * p^x * (1 - p)^(n - x)
emp = plot(f, 0:n, line=:stem, marker=:auto, label="Empirical")
theo = plot(bi, 0:n, line=:stem, marker=:circle, label="Theoretical", color=:red)
plot(emp, theo)