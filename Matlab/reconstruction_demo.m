%% Run Reconstruction for Spectral DiffuserCam
% Last update: 9/29/2020

addpath('helper_functions/')  
addpath('SampleData/')

%% Load in Calibration data and PSF
load('calibration.mat')
wavelengths = wavs;

im = double(imread('meas_dice.png'));

%% Pre-process images and data 
% crop all data to the valid mask-pixels 
c1 = 100; c2 = 419;
c3= 80; c4=539;

mask = mask(c1:c2, c3:c4,:);
psf = psf(c1:c2, c3:c4);
im = double(im(c1:c2, c3:c4));

% normalize PSF 
psf = psf/norm(psf, 'fro');

% subtract pixel defect from mask and image 
mask_sum = sum(mask, 3);
[maxval,idx]=max(mask_sum(:));
[row, col] = ind2sub(size(mask_sum), idx);
mask(row-2:row+2, col-2:col+2, :)= 0;

im = im/max(max(im));
im(row-2:row+2, col-2:col+2, :)= 0;


%% Put everything on GPU (if using GPU) 
opts.use_gpu = 1;
if opts.use_gpu 
    psf = gpuArray(single(psf(:,1:end)));
    if mod(size(mask,3),2) == 0
        mask = gpuArray(single(mask(:,1:end, 1:end)));
    else
        mask = gpuArray(single(mask(:,1:end, 1:end-1)));
        wavelengths = wavelengths(1:end-1);
    end
    im = gpuArray(single(im));
end


%% Define reconstruction options (leave these alone for defaults) 
name='thordog';
opts.fista_iters =500;     % Number of FISTA iterations
opts.denoise_method = 'tv'; % options: 'tv', 'non-neg', 'native', 'tv_lowrank'

opts.tv_lambda =   .003;    % TV tuning parameter (higher is more TV)
opts.tv_lambday =  1;       % TV tuning parameter in y (compared to x)
opts.tv_lambdaw =  .01;     % TV tuning parameter in lambda (compared to x)
opts.lowrank_lambda = .00005; % Tuning parameter for the low-rank constraint

opts.display_every = 1;      % how often to display the reconstruction 
opts.save_data = 1;          % save the intermediate recon images
opts.save_data_freq = 100;   % save data every 50 iterations

% Filename to save the results
filename_save = sprintf('saved_recons/%s_%s_lambda_%f_iterations_%f_', name, opts.denoise_method, opts.tv_lambda, opts.fista_iters);
opts.save_data_path = filename_save;


%% Run inverse solver
[xout, loss_list] = fista_spectral_3d(im, psf, mask, opts);


%% 
xout=fliplr(flipud(gather(xout)));
false_color = false_color_function(xout);
figure(), imshow(false_color); title('False-color reconstruction')

figure(), imshow3D(xout);