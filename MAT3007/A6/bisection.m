function [x, fx, iter] = bisection(f, fp, l, r, e)

    % minimize function f using bisection method.
    % args: f : function handle
    %       fp: function derivative
    %       l : left endpoint
    %       r : right endpoint
    %       e : max error tolerance
    
    iter = 0;
    
    if (l > r)
        fprintf("error: l > r\n")
        return
    end
    
    if (fp(l) > 0 || fp(r) < 0)
        fprintf("error: f not convex")
        return
    end
    
    while (r - l > e)
        iter    = iter + 1;
        m       = (r + l) / 2;
        %fprintf("iter:%02d    left:%f	right:%f	optval:%f\n", iter, l, r, f(m));
        
        if (fp(m) > 0)
            r = m;
        else
            l = m;
        end
    end

    x       = (r + l) / 2;
    fx		= f(x);
end