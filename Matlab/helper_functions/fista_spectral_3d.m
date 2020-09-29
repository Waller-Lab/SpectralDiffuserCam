function [xout, loss_list]=fista_spectral_3d(input_image, psf, mask, opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FISTA implementation for Spectral DiffuserCam
% Last update: 9/29/2020
%
%
% Inputs
% input_image ............. input, measured image 
% psf ..................... PSF, same size as input image
% opts..................... options file
%  opts.fista_iters ....... Number of FISTA iterations
%  opts.denoise_method .... Either 'non-neg' or 'tv'
%  opts.tv_lambda ......... amount of tv
%  opts.tv_iters .......... number of inner-loop iterations 
%
% Outputs
% xout .................... deblurred image
% loss_list ............... list of losses 


figure(2020)

[Ny, Nx, n_filters] = size(mask);   %Get problem size

% Setup convolutional forward op
p1 = floor(Ny/2);
p2 = floor(Nx/2);
pad2d = @(x)padarray(x,[p1,p2],'both');  %2D padding
crop2d = @(x)x(p1+1:end-p1,p2+1:end-p2,:); %2D cropping

vec = @(X)reshape(X,numel(X),1);
Hs = fftn(ifftshift(pad2d(psf)));  %Compute 3D spectrum
Hs_conj = conj(Hs);

Hfor_modified = @(x)pad2d(sum(mask.*crop2d(real((ifftn(Hs.*fftn((x)))))),3));

Hfor = @(x)sum(mask.*crop2d(real((ifftn(Hs.*fftn((x)))))),3);
Hadj = @(x)real((ifftn(Hs_conj.*fftn((pad2d(repmat(x, [1,1,size(mask, 3)]).*mask))))));

maxeig = power_iteration(Hfor_modified, pad2d(psf), 10);
L = maxeig*45;


lambda = opts.tv_lambda;

% TV denoising options: 
l = 0;  u = Inf;
clear parsin
parsin.MAXITER=opts.tv_iters;
parsin.epsilon=1e-5;
parsin.print=0;
parsin.tv='iso';
parsin.use_gpu = opts.use_gpu;


if strcmp(opts.denoise_method, 'tv') == 1
    prox = @(x)(1/2*(max(x,0) + (tv3dApproxHaar(x, opts.tv_lambda/L, opts.tv_lambday, opts.tv_lambdaw))));
    loss = @(err, x) norm(err,'fro')^2 + 2*lambda/L*tlv(x, parsin.tv, parsin.use_gpu);
elseif strcmp(opts.denoise_method, 'tv_lowrank') == 1
    prox = @(x)(1/3*(max(x,0) + tv3dApproxHaar(x, opts.tv_lambda/L, opts.tv_lambday, opts.tv_lambdaw)+ soft_thresh_lowrank(x, opts.lowrank_lambda)));
    loss = @(err, x) norm(err,'fro')^2 + 2*lambda/L*tlv(x, parsin.tv, parsin.use_gpu);
elseif strcmp(opts.denoise_method, 'native') == 1
    prox = @(x) (1/2 * (max(x,0) + soft_thresh(x, opts.tv_lambda/L)));
    loss = @ (err, x) norm(err,'fro')^2;
elseif strcmp(opts.denoise_method, 'non-neg') == 1
    prox = @(x)max(x,0);
    loss = @ (err, x) norm(err,'fro')^2;  
end


if opts.save_data == 1
    if ~exist(opts.save_data_path, 'dir')
       mkdir(opts.save_data_path)
    end
    y_save = gather(input_image);
    mask_save = gather(mask);
    psf_save = gather(psf);
    
 filename = sprintf('%s/params.mat', opts.save_data_path);
 save(filename, 'y_save', 'psf_save', 'mask_save', 'opts', 'L');
end

padded_input = pad2d(input_image);

%% Start FISTA 
xk = zeros(Ny*2, Nx*2);
vk = zeros(Ny*2, Nx*2);
tk = 1.0;

loss_list = [];

for i=1:opts.fista_iters
    xold = xk;
    vold = vk;
    told = tk;
    
    error = Hfor(vold) - input_image;
    grads = Hadj(error);
    
    xk = prox(vold - 1/L*grads);
    
    tk = 1 + sqrt(1+4*told^2)/2;
    vk = xk + (told-1)/tk *(xk- xold);
    
    loss_i = loss(error, xk);
    loss_list = [loss_list, loss_i];
   
    
    if mod(i,opts.display_every) == 0 
        subplot(1,4,1)
        
        x_print = gather(xk);
        x_print = fliplr(flipud((x_print)));
        false_color = false_color_function(x_print);
        %imagesc(reshape(sum(xk, 3), Ny*2, Nx*2));
        imagesc(false_color);
        title('False color image')

        subplot(1,4,2)
        imagesc(reshape(sum(x_print, 2), Ny*2, n_filters));
        title('Y-Lambda sum projection')

        subplot(1,4,3)
        imagesc(reshape(sum(x_print, 1), Nx*2, n_filters));
        title('X-Lambda sum projection')
    
        subplot(1,4,4),
        semilogy(loss_list);
        title('Loss')
        drawnow
    end
    
    if mod(i, opts.save_data_freq) ==0 && opts.save_data == 1
        opts.save_data_path
        
        filename = sprintf('%s/iter%i.mat', opts.save_data_path, i);
        x_save = gather(xk);
        save(filename, 'x_save');
    end
    
end

xout = crop2d(xk);

end
