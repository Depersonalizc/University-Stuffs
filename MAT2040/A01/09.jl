# dices rolling
using Plots

x = 2:12
obs = 10000
outcomes = Dict((i, 0) for i in x)

for _ in 1:obs
    s = sum(rand(1:6, 2))
    outcomes[s] += 1
end

f(x) = outcomes[x] / obs
scatter(x, f.(x), title="empirical probabilities", legend=false)
