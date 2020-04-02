# MAT2040 Linear Algebra
# Homework 2 Question 20
# Script by MAO Licheng

include("q20data.jl")
using PyPlot


# belows are for (a)

ax1 = plt[:subplot](111)
ax1[:bar](1:length(ng[1][1,:]), ng[1][1,:])
ax1[:set_xticks]([i for i in 1:length(alphabet)])
ax1[:set_xticklabels]([string(i) for i in alphabet])
ax1[:set_xlabel]("letter")
ax1[:set_ylabel]("empirical PMF")
ax1[:set_title]("Letter PMF, estimates from Ulysses text")

# the above code may be platform dependent, if it does not work, try the following version.
#
# ax1 = plt.subplot(111)
# ax1.bar(1:length(ng[1][1,:]), ng[1][1,:])
# ax1.set_xticks([i for i in 1:length(alphabet)])
# ax1.set_xticklabels([string(i) for i in alphabet])
# ax1.set_xlabel("letter")
# ax1.set_ylabel("empirical PMF")
# ax1.set_title("Letter PMF, estimates from Ulysses text")


# belows are for (b)

using3D()
ax1 = plt[:subplot](111, projection="3d")
ax1[:bar3d](
    reshape([i for i in 1:size(ng[2])[1], j in 1:size(ng[2])[2]],:), 
    reshape([j for i in 1:size(ng[2])[1], j in 1:size(ng[2])[2]],:),
    reshape(fill(0, size(ng[2])[1], size(ng[2])[2]),:),
    0.7,
    0.7,
    reshape(ng[2],:)
)
ax1[:set_xticks]([i for i in 1:length(alphabet)])
ax1[:set_yticks]([i for i in 1:length(alphabet)])
ax1[:set_xticklabels]([string(i) for i in alphabet])
ax1[:set_yticklabels]([string(i) for i in alphabet])
ax1[:set_xlabel]("current letter")
ax1[:set_ylabel]("next letter")
ax1[:set_zlabel]("emperical conditional PMF")
ax1[:view_init](elev=60,azim=325)
ax1[:set_title]("Conditional PMF of next letter given current letter")

# the above code may be platform dependent, if it does not work, try the following version.
#
# using3D()
# ax1 = plt.subplot(111, projection="3d")
# ax1.bar3d(
#     reshape([i for i in 1:size(ng[2])[1], j in 1:size(ng[2])[2]],:), 
#    reshape([j for i in 1:size(ng[2])[1], j in 1:size(ng[2])[2]],:),
#    reshape(fill(0, size(ng[2])[1], size(ng[2])[2]),:),
#    0.7,
#    0.7,
#    reshape(ng[2],:)
#)
# ax1.set_xticks([i for i in 1:length(alphabet)])
# ax1.set_yticks([i for i in 1:length(alphabet)])
# ax1.set_xticklabels([string(i) for i in alphabet])
# ax1.set_yticklabels([string(i) for i in alphabet])
# ax1.set_xlabel("current letter")
# ax1.set_ylabel("next letter")
# ax1.set_zlabel("emperical conditional PMF")
# ax1.view_init(elev=60,azim=325)
# ax1.set_title("Conditional PMF of next letter given current letter")


# belows are for (c)
function place(string,alphabet)
    # Taken from: 
    # http://www.cs.duke.edu/courses/spring09/cps111/notes/markovChains.pdf
    # Original version in MATLAB
    na = length(alphabet); 
    n = length(string);
    pos = 0;
    for k = 1:n
        i = findall(x->x==string[k], alphabet)
        if isempty(i)
            i = [1]
        end
        pos = pos * na + i[1] - 1;
    end
    # Convert to Matlab style array indexing (so minimum pos is 1, not 0)
    pos = pos + 1;
    return pos
end

function draw(y, p)
    # Taken from: 
    # http://www.cs.duke.edu/courses/spring09/cps111/notes/markovChains.pdf
    # Original version in MATLAB
    c = cumsum(p);
    c = c ./ c[end];
    c = [0; c]; # prepand 0
    K = length(c); # num_possibilities + 1
    r = rand(1, 1);
    r = r[:]
    v = zeros(1, 1);
    for i = 2:K
        # find which char is match, return that
        if r[1] <= c[i] && r[1] > c[i - 1]
            return y[i-1]
        end
    end
end

function randomSentence(ng, alphabet, n, order)
    # prepare blank string
    s = fill(' ', (1, n))
    for k = 1:n
        j = max(1, k - order)
        pp = place(s[j:k-1], alphabet)
        g = min(k, order + 1)
        global s[k] = alphabet[draw(1:length(alphabet), ng[g][pp, :])]
    end
    return String(s[:])
end


randomSentence(ng, alphabet, 100, 0)

randomSentence(ng, alphabet, 100, 1)

randomSentence(ng, alphabet, 100, 2)

[1/27 for _ in 1:27]