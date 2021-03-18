# Machine Learning

## Files

### cycle.py
Contains the code to run the cycle_gan. You will need to provide your own data for this. You can modify the two data loaders. Currently they are set up to point at some files on my local machine.

### Autoencoder.py
This file contains different Machine Learning models that I have set up. I have tried to make them flexible to allow for different sizes of images.

- Autoencoder
  - Shrinks down an input image to a latent dimension / vector and then generates it back up
  - if input == output, can be used to denoise images, compress files etc.
- Encoder
  - Transforms an input image to a vector size, this can be used for feature extraction
- Decoder
  - Take a latent vector and builds it up to a specfic output size
- Discriminator
  - Encoder with output of size 1 and a sigmoid activation function
- SmoothAutoencoder
  - This allows you to learn the image by chunking it up into different size pieces, it uses the autoencoder on the individual pieces and then uses some conv2d's to try and smooth the stitched together pieces.