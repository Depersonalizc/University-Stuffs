function v = blk_stack(V, bsz)
    
    % block-stack matrix V into column vector v, Psi style.
    % args: V: matrix to bock-stack
    %       bsz: block size

    [~, n]  = size(V);
    vect    = @(b) b.data(:);

    V = blockproc(V, [bsz bsz], vect);
    v = blockproc(V, [bsz^2 n/bsz], vect);
    
end
