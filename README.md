# Spectral DiffuserCam - lensless snapshot hyperspectral imaging

### [Project Page](https://waller-lab.github.io/SpectralDiffuserCam/) | [Paper](https://www.osapublishing.org/optica/abstract.cfm?uri=optica-7-10-1298)


Please cite the following paper when using this code:
[Spectral DiffuserCam: lensless snapshot hyperspectral imaging with a spectral filter array] (https://www.osapublishing.org/optica/abstract.cfm?uri=optica-7-10-1298)

Kristina Monakhova, Kyrollos Yanny, Neerja Aggarwal, and Laura Waller

## Contents

1. [Data] Data
2. [Setup] Setup 
3. [Description] Description 

## Data
Sample data (needed to run the code) can be found [here](https://drive.google.com/drive/folders/1dmfzkTLFZZFUYW8GC6Vn6SOuZiZq47SS?usp=sharing)

Please place the downloaded data in SampleData folder the Python and/or Matlab folders.

## Setup
Clone this project using: s
```
git clone https://github.com/Waller-Lab/SpectralDiffuserCam.git
```

The dependencies can be installed by using:
```
conda env create -f environment.yml
source activate spectral_diffusercam
```

## Description 
This repository contains code in both Python and Matlab that is needed to process raw Spectral DiffuserCam images and reconstruct 3D hyperspectral volumes from the raw 2D measurements.  Four example raw images are provided, along with the calibrated point spread function and spectral filter function. Both the Python and Matlab versions support GPU acceleration. In Python, this is accomplished using cupy.  