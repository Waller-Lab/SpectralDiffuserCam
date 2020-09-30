import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from IPython.core.display import display, HTML
from ipywidgets import interact, widgets, fixed

import sys
sys.path.append('helper_functions/')
   
    
def plotf2(r, img, ttl, sz):
    #fig = plt.figure(figsize=(2, 2));
    #plt.figure(figsize=(20, 20));
    plt.title(ttl+' {}'.format(r))
    plt.imshow(img[:,:,r], cmap="gray", vmin = 0, vmax = np.max(img));
    plt.axis('off');
    fig = plt.gcf()
    fig.set_size_inches(sz)
    plt.show();
    #display(fig)
    #clear_output(wait=True)
    return 

def plt3D(img, title = '', size = (5,5)):
    #fig = plt.figure(figsize=sz);
    interact(plotf2, 
             r=widgets.IntSlider(min=0,max=np.shape(img)[-1]-1,step=1,value=1), 
             img = fixed(img), 
             continuous_update= False, 
             ttl = fixed(title), 
             sz = fixed(size));
    
def crop(x):
    DIMS0 = x.shape[0]//2  # Image Dimensions
    DIMS1 = x.shape[1]//2  # Image Dimensions

    PAD_SIZE0 = int((DIMS0)//2)                           # Pad size
    PAD_SIZE1 = int((DIMS1)//2)                           # Pad size

    C01 = PAD_SIZE0; C02 = PAD_SIZE0 + DIMS0              # Crop indices 
    C11 = PAD_SIZE1; C12 = PAD_SIZE1 + DIMS1              # Crop indices 
    return x[C01:C02, C11:C12,:]

def pre_plot(x):
    x = np.fliplr(np.flipud(x))
    x = x/np.max(x)
    x = np.clip(x, 0,1)
    return x
    
    
def stack_rgb_opt(reflArray, opt = 'helper_functions/false_color_calib.mat', scaling = [1,1,2.5]):
    
    color_dict = scipy.io.loadmat(opt)
    red = color_dict['red']; green = color_dict['green']; blue = color_dict['blue']
    
    reflArray = reflArray/np.max(reflArray)
    
    red_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    green_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    blue_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    
    for i in range(0,64):
        red_channel = red_channel + reflArray[:,:,i]*red[0,i]*scaling[0]
        green_channel = green_channel + reflArray[:,:,i]*green[0,i]*scaling[1]
        blue_channel = blue_channel + reflArray[:,:,i]*blue[0,i]*scaling[2]
        
    red_channel = red_channel/64.
    green_channel = green_channel/64.
    blue_channel = blue_channel/64.

    stackedRGB = np.stack((red_channel,green_channel,blue_channel),axis=2)

    return stackedRGB

def preprocess(mask, psf, im):
    
    # Crop indices
    c1 = 100; c2 = 420; c3 = 80; c4 = 540
    
    # Crop and normalize mask
    mask = mask[c1:c2, c3:c4, :] 
    mask = mask/np.max(mask)
    
    # Crop and normalize PSF
    psf = psf[c1:c2, c3:c4]
    psf = psf/np.linalg.norm(psf)

    # Remove defective pixels in mask calibration 
    mask_sum = np.sum(mask, 2)
    ind = np.unravel_index((np.argmax(mask_sum, axis = None)), mask_sum.shape)
    mask[ind[0]-2:ind[0]+2, ind[1]-2:ind[1]+2, :] = 0
    
    # Remove defective pixels in measurement 
    im = im[c1:c2, c3:c4]
    im = im/np.max(im)
    im[ind[0]-2:ind[0]+2, ind[1]-2:ind[1]+2] = 0
    return mask, psf, im