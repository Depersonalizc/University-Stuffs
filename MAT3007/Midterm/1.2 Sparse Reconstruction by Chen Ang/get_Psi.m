function Psi = get_Psi(m,n,dsz)           
    
    % fix dct-block size to 8
    if nargin <= 2
        dsz     = 8;
    end
    
    % display error if the image size is not compatible with the dct-block
    % size
    if mod(m,dsz) > 0 || mod(n,dsz) > 0
        error(strcat('Image size not a multiple of dsz = ',num2str(dsz,'%i')));
        Psi = [];
        return
    end
    
    % build Psi 
    D           = dctmtx(dsz); 
    Bdct        = kron(D',D);
    
    sz          = (m/dsz)*(n/dsz);
    Psi         = kron(speye(sz),Bdct);
end