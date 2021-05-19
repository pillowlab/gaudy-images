# gaudy-images
This is the repository code for "High-contrast 'gaudy' images improve the training of deep neural network models of visual cortex" by Cowley and Pillow, NeurIPS 2020.

Link to paper: <a href="https://proceedings.neurips.cc//paper/2020/hash/f610a13de080fb8df6cf972fc01ad93f-Abstract.html" target="_blank">NeurIPS abstract</a>

* We want to predict neural responses of higher-order visual cortex (e.g., V4, IT) from natural images. 
* We use a deep neural network to make this prediction, which require a *lot* of data to train accurately---data we don't have in neuroscience. 
* We reduce the amount of training data required by using high-contrast, binarized *gaudy images*.
* Training on *gaudy images* yields better DNN prediction than training on the same number of normal images.
* In our paper, we find this is because gaudy images overemphasize edges in the image.


# gaudy transformation
To transform a normal, colorful image into its gaudy version:

<img src="/extra/gaudy_transformation.png" width="741" height="221">

In code (three lines!):

```python

img = download_image()  # download example img with shape 224 x 224 x 3
img_gaudy = np.copy(img)

mean_img = np.mean(img)  # take mean across all pixels and channels
img_gaudy[img_gaudy < mean_img] = 0   # transform each pixel to the rails
img_gaudy[img_gaudy >= mean_img] = 255

plt.imshow(img)
plt.imshow(img_gaudy)
```

# this repository
This repository contains code that was used to generate figures in our paper.
We use Python XX with tensorflow XX and keras XX.

*Warning:* This is research code, not production code. The best use is to see how 
some code works, and then copy/paste/steal it for your own use. 
*Warning:* We do not include our image dataset due to memory constraints/permissions. We use a subset of 
images from the <a href="https://yahooresearch.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images-for" target="_blank">Yahoo/FlickR image set (YFCC100M)</a>, which is freely available. You could also use ImageNet.

We utilize deep neural networks already trained on ImageNet. These DNNs are available within Keras.

Code for each figure is stored in separate folders (e.g., fig1, fig2, ...).

If you have a question with the repository, please raise an issue through github on this project.







