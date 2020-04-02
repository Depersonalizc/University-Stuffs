using LinearAlgebra
U=rand(4,4); V=rand(4,4); b=ones(4,1);
V
rank(V)
ϵ₄E = inv(U) # change-of-basis matrix from ϵ₄ to E
ϵ₄F = inv(V) # change-of-basis matirx from ϵ₄ to F
c = ϵ₄E * b # representation of b w.r.t. E
d = ϵ₄F * b # representation of b w.r.t. F
norm(b-U*c) 
norm(b-V*d)

S = EF = ϵ₄F*U
T = FE = ϵ₄E*V
norm(d-S*c)
norm(c-T*d)