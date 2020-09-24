function [x, fx, iter] = ausection(f, l, r, e)

    % minimize function f using golden section method.
    % args: f: function handle
    %       l: left endpoint
    %       r: right endpoint
    %       e: max error tolerance
    
    PHI     = (3 - sqrt(5)) / 2;
    iter    = 0;
    
    if (l > r)
        fprintf("error: l > r\n")
        return
    end
    
    nl  = (1 - PHI) * l + PHI * r;
    nr  = (1 - PHI) * r + PHI * l;
    fnl = f(nl);
    fnr = f(nr);
    
    while (r - l > e)
        iter    = iter + 1;
        %fprintf("iter:%02d    left:%f	right:%f	optval:%f\n", iter, l, r, f((r+l) / 2) );
        
        if (fnl > fnr)
            l   = nl;
            nl  = nr;
            fnl = fnr;
            nr  = (1 - PHI) * r + PHI * l;
            fnr = f(nr);
        else
            r   = nr;
            nr  = nl;
            fnr = fnl;
            nl  = (1 - PHI) * l + PHI * r;
            fnl = f(nl);
        end
    end
    
    x       = (r + l) / 2;
    fx		= f(x);
end