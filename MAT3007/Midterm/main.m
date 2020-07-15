tic;
% setup
zoom    = 1;
U       = imread('.\test_images\512_512_lena.png');
%U       = rgb2gray(U);
U       = imresize(U, zoom);
U       = double(U) / 255;
[m, n]  = size(U);

Ind     = imread('.\test_masks\512_512_random90.png');
Ind     = logical(ceil(Ind / 255));
Ind     = imresize(Ind, zoom);
s       = sum(Ind, 'all');
delta   = 0.06;

bsz = 8;
Psi = get_Psi(m, n, bsz);

% stacking U and Ind in Psi style
u   = blk_stack(U, bsz);
ind = blk_stack(Ind, bsz);

% form A and b
i       = 1:s;
j       = zeros(1,s);
count   = 1;
for col = 1:m*n
    if ind(col) == 1
        j(count) = col;
        count = count + 1;
    end
end
A = sparse(i, j, ones(1, s), s, m*n);
b = A * u;

% linprog
% min(c'x) s.t. Mx <= d
del     = delta*ones(s, 1);
I       = speye(m*n);
ze      = sparse(s, m*n);
c       = [zeros(m*n, 1); ones(m*n, 1)];
M       = [-Psi -I; Psi -I; -A ze; A ze];
d       = [zeros(2*m*n, 1); del-b; del+b];
options = optimoptions('linprog', 'Algorithm', 'interior-point', 'ConstraintTolerance', 1e-3, 'Display', 'iter');
x       = linprog(c, M, d, [], [], [], [], options);

% scale and transform the stacked 
% image back into matrix
x = uint8(x(1:m*n) / 255);
X = blk_unstack(x, bsz);
psnr = PSNR((U*255), double(X));
imshow(X);
toc