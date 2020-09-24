% main program
tic;

% setup
U       = imread('.\test_images\512_512_lena.png');
if size(U, 3) == 3
    U 	= rgb2gray(U);
end
U       = double(U) / 255;
[m, n]  = size(U);

Ind     = imread('.\test_masks\512_512_random50.png');
Ind     = logical(ceil(Ind / 255));
s       = sum(Ind, 'all');
delta   = 0.0006;

% block-stack U and Ind in Psi style
bsz = 8;
Psi = get_Psi(m, n, bsz);
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

s = m*n
% linprog
% min(c'x) s.t. Mx <= d
del     = delta*ones(s, 1);
I       = speye(m*n);
ze      = sparse(s, m*n);
c       = [zeros(m*n, 1); ones(m*n, 1)];
M       = [-Psi -I; Psi -I; -A ze; A ze];
d       = [zeros(2*m*n, 1); del-b; del+b];
options = optimoptions('linprog', 'Algorithm', 'interior-point', ...
                       'ConstraintTolerance', 1e-3, ...
                       'Display', 'iter');
x       = linprog(c, M, d, [], [], [], [], options);

% scale and transform the stacked image back into matrix
x = uint8( x(1:m*n)*255 );
X = blk_unstack(x, bsz);

% evaluate PSNR
psnr = PSNR((U*255), double(X));
imshow(X);

toc