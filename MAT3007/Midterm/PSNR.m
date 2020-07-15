function psnr = PSNR(origImg, recImg)

    % compute PSNR of the reconstructed image.
    % args: origImg: original image
    %       recImg: reconstructed image
    
    [m, n]  = size(origImg);
    x       = reshape(recImg, [], 1) / 255;
    u       = reshape(origImg, [], 1) / 255;
    nrm     = norm(x - u);
    psnr    = 10 * log(m*n / nrm^2) / log(10);

end

