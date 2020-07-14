function v = blk_stack(V, bsz)

    [~, n]  = size(V);
    vect    = @(b) vec(b.data);

    V = blockproc(V, [bsz bsz], vect);
    v = blockproc(V, [bsz^2 n/bsz], vect);
    
end
