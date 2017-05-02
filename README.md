This is the source code and pretrained model for the webcam pix2pix demo I posted recently on [twitter](https://twitter.com/memotv/status/858397873712623616) and vimeo. It uses deep learning, or to throw in a few buzzwords: *deep convolutional conditional generative adversarial network autoencoder*. 


[![video 1](https://cloud.githubusercontent.com/assets/144230/25585045/9b932e50-2e90-11e7-9bb2-692ef9629f0a.png)
*video 1*
](https://vimeo.com/215339817)

[![video 2](https://cloud.githubusercontent.com/assets/144230/25584635/b67b0bea-2e8e-11e7-8b12-f8356241728b.png)
*video 2*
](https://vimeo.com/215514169)



# Overview
The code in this particular repo actually has nothing to do with pix2pix, GANs or even deep learning. It just loads *any* pre-trained tensorflow model (as long as it complies with a few constraints), feeds it a processed webcam input, and displays the output of the model. It just so happens that the model I trained and used is pix2pix (details below). 

I.e. The steps can be summarised as:

1. Collect data: scrape the web for a ton of images, preprocess and prepare training data
2. Train and export a model
3. Preprocessing and prediction: load pretrained model, feed it live preprocessed webcam input, display the results. 

# 1. Data
I scraped art collections from around the world from the [Google Art Project on wikimedia](https://commons.wikimedia.org/wiki/Category:Google_Art_Project_works_by_collection). A **lot** of the images are classical portraits of rich white dudes, so I only used about 150 collections, trying to keep the data as geographically and culturally diverse as possible (full list I used is [here](./gart_canny_256_info/collections.txt)). But the data is still very euro-centric, as there might be hundreds or thousands of scans from a single European museum, but only 8 scans  from an Arab museum. 

I downloaded the 300px versions of the images, and ran a batch process to :

- Rescale them to 256x256 (without preserving aspect ratio)
- Run a a simple edge detection filter (opencv canny)

I also ran a batch process to take multiple crops from the images (instead of a non-uniform resizing) but I haven't trained on that yet. Instead of canny edge detection, I also started looking into the much better  'Holistically-Nested Edge Detection' (aka [HED](https://github.com/s9xie/hed)) by Xie and Tu (as used by the original pix2pix paper), but haven't trained on that yet either. 

This is done by the [preprocess.py](preprocess.py) script (sorry no command line arguments, edit the script to change paths and settings, should be quite self-explanatory).


**A small sample of the training data - including predictions of the trained model - can be seen [here](http://memo.tv/gart_canny_256_pix2pix/).**
Right-most column is the original image, left-most column is the preprocessed version. These two images are fed into the pix2pix network as a 'pair' to be trained on. The middle column is what the model learns to produce *given only the left-most column*. (The images show each training iteration - i.e. the number on the left, which goes from 20,000 to 58,000, so it gradually gets better the further down you go on the page). 

[![training_data](https://cloud.githubusercontent.com/assets/144230/25617554/bd2f3c16-2f3a-11e7-9e25-75792fbc3380.png)](http://memo.tv/gart_canny_256_pix2pix/)


I also trained an unconditional GAN (i.e. normal [DCGAN](https://github.com/Newmu/dcgan_code) on this same training data. An example of its output can be seen below. (This is generating 'completely random' images that resemble the training data). 

![dcgan](https://cloud.githubusercontent.com/assets/144230/25617262/58c9dc46-2f39-11e7-97b9-d546cc6cc00c.png)


# 2. Training
The training and architecture is straight up '*Image-to-Image Translation with Conditional Adversarial Nets*' by Isola et al (aka [pix2pix](https://phillipi.github.io/pix2pix/)). I trained with the [tensorflow port](https://github.com/affinelayer/pix2pix-tensorflow) by @affinelayer (Christopher Hesse), which is also what's powering that '[sketch-to-cat](https://affinelayer.com/pixsrv/)'- demo that went viral recently. He also wrote a nice [tutorial](https://affinelayer.com/pix2pix/) on how pix2pix works. Infinite thanks to the authors (and everyone they built on) for making their code open-source!

I only made one infinitesimally tiny change to the tensorflow-pix2pix training code, and that is to add *tf.Identity* to the generator inputs and outputs with a human-readable name, so that I can feed and fetch the tensors with ease. **So if you wanted to use your own models with this application, you'd need to do the same**. (Or make a note of the input/output tensor names, and modify the json accordingly, more on this below). 

**You can download my pretrained model from the [Releases tab](https://github.com/memo/webcam-pix2pix-tensorflow/releases).**

![pix2pix_diff](https://cloud.githubusercontent.com/assets/144230/25583118/4e4f9794-2e88-11e7-8762-889e4113d0b8.png)

# 3. Preprocessing and prediction
What this particular application does is load the pretrained model, do live preprocessing of a webcam input, and feed it to the model. I do the preprocessing with old fashioned basic computer vision, using opencv. It's really very minimal and basic. You can see the GUI below (the GUI uses [pyqtgraph](http://www.pyqtgraph.org/)).

![ruby](https://cloud.githubusercontent.com/assets/144230/25586317/b3f4e65e-2e96-11e7-809d-5a6296d2ed64.png)

Different scenes require different settings.

E.g. for 'live action' I found **canny** to provide better (IMHO) results, and it's what I used in the first video at the top. The thresholds (canny_t1, canny_t2) depend on the scene, amount of detail, and the desired look. 

If you have a lot of noise in your image you may want to add a tiny bit of **pre_blur** or **pre_median**. Or play with them for 'artistic effect'. E.g. In the first video, at around 1:05-1:40, I add a ton of median (values around 30-50).

For drawing scenes (e.g. second video) I found **adaptive threshold** to give more interesting results than canny (i.e. disable canny and enable adaptive threshold), though you may disagree. 

For a completely *static input* (i.e. if you **freeze** the capture, disabling the camera update) the output is likely to flicker a very small amount as the model makes different predictions for the same input - though this is usually quite subtle. However for a *live* camera feed, the noise in the input is likely to create lots of flickering in the output, especially due to the high susceptibility of canny or adaptive threshold to noise, so some temporal blurring can help. 

**accum_w1** and **accum_w2** are for temporal blurring of the input, before going into the model:
new_image = old_image * w1 + new_image * w2 (so ideally they should add up to one - or close to). 

**Prediction.pre_time_lerp** and **post_time_lerp** also do temporal smoothing:
new_image = old_image * xxx_lerp + new_image * (1 - xxx_lerp)
pre_time_lerp is before going into the model, and post_time_lerp is after coming out of the model. 

Zero for any of the temporal blurs disables them. Values for these depend on your taste. For both of the videos above I had all of pre_model blurs (i.e. accum_w1, accum_w2 and pre_time_lerp)  set to zero, and played with different post_time_lerp settings ranging from 0.0 (very flickery and flashing) to 0.9 (very slow and fadey and 'dreamy'). Usually around 0.5-0.8 is my favourite range. 

# Using other models
If you'd like to use a different model, you need to setup a JSON file similar to the one below. 
The motivation here is that I actually have a bunch of JSONs in my app/models folder which I can dynamically scan and reload, and the model data is stored elsewhere on other disks, and the app can load and swap between models at runtime and scale inputs/outputs etc automatically. 

	{
		"name" : "gart_canny_256", # name of the model (for GUI)
		"ckpt_path" : "./models/gart_canny_256", # path to saved model (meta + checkpoints). Loads latest if points to a folder, otherwise loads specific checkpoint
		"input" : { # info for input tensor
			"shape" : [256, 256, 3],  # expected shape (height, width, channels) EXCLUDING batch (assumes additional axis==0 will contain batch)
			"range" : [-1.0, 1.0], # expected range of values 
			"opname" : "generator/generator_inputs" # name of tensor (':0' is appended in code)
		},
		"output" : { # info for output tensor
			"shape" : [256, 256, 3], # shape that is output (height, width, channels) EXCLUDING batch (assumes additional axis==0 will contain batch)
			"range" : [-1.0, 1.0], # value range that is output
			"opname" : "generator/generator_outputs" # name of tensor (':0' is appended in code)
		}
	}


# Requirements
- python 2.7 (likely to work with 3.x as well)
- tensorflow 1.0+
- opencv 3+ (probably works with 2.4+ as well)
- pyqtgraph (only tested with 0.10)

Tested only on Ubuntu 16.04, but should work on other platforms. 

I use the Anaconda python distribution which comes with almost everything you need, then it's (hopefully) as simple as:
1. Download and install anaconda from https://www.continuum.io/downloads
2. Install tensorflow https://www.tensorflow.org/install/ (Which - if you have anaconda - is often quite straight forward since most dependencies are included)
3. Install opencv and pyqtgraph

	conda install -c menpo opencv3
	conda install pyqtgraph
    
    
    
# Acknowledgements
Infinite thanks once again to

* Isola et al for [pix2pix](https://phillipi.github.io/pix2pix/) and @affinelayer (Christopher Hesse) for the [tensorflow port](https://github.com/affinelayer/pix2pix-tensorflow)
* Radford et al for [DCGAN](https://github.com/Newmu/dcgan_code) and @carpedm20 (Taehoon Kim) for the [tensorflow port](https://github.com/carpedm20/DCGAN-tensorflow)
* The [tensorflow](https://www.tensorflow.org/) team
* Countless others who have contributed to the above, either directly or indirectly, or opensourced their own research making the above possible
* My [wife](http://janelaurie.com/) for putting up with me working on a bank holiday to clean up my code and upload this repo. 