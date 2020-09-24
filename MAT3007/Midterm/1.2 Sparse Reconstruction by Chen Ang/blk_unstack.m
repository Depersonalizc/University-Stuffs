function V = blk_unstack(v, bsz)
    
    % unstack column vector v into matrix V, Psi style.
    % args: v: vector to unstack
    %       bsz: block size

    [m, ~]  = size(v);
    m       = m / bsz^2;
    m       = sqrt(m);
    
    v = blockproc(v, [m*bsz^2 1], @(b) reshape(b.data', bsz^2, []));
    V = blockproc(v, [bsz^2 1], @(b) reshape(b.data', bsz, []));

end