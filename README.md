# Generative Fill Tensorflow & Keras


<p float="left">
  <img src="results/ema_epoch_50.jpeg" width="800" height="500"> 
</p>


This work uses diffusion models to implement generative fill techniques like image unmasking, inpainting, expansion and 
various generative fill techniques.<br>

**Image unmasking**<br>
This model's forward diffusion is the same as that of the basic diffusion model; 
the difference lies in the reverse diffusion phase, where along with the noised version of the image, and the 
timestep we also feed the masked image to the model, and the model is made to predict the noise in the noised image 
using the masked image as a guidance.
<br>

During generation, we feed the three inputs (noised version of image, time step, masked image) to the model to 
predict & remove the noise throughout the diffusion timesteps.
<br>

[Check out my DDPM implementation.](https://github.com/NITHISHM2410/diffusion-model-tf-ddpm)

## Target
 * [x] Unmask the edges of images.<br>
 * [ ] Reconstructing different areas of images not just the edges a.k.a image inpainting.<br>
 * [ ] Implement other generative fill techniques.<br>
 * [ ] Generative filling in latent space.<br>
 * [ ] Applying the same high resolution images.<br>

## Acknowledgements

 - [LHQ Dataset](https://universome.github.io/alis)
 - [DDPM Official](https://github.com/hojonathanho/diffusion)


## Training & Generation demo

- [GenerativeFill](https://www.kaggle.com/code/nithishm2410/generativefill)









