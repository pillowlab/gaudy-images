# gaudy-images
This is the repository code for "High-contrast 'gaudy' images improve the training of deep neural network models of visual cortex" by Cowley and Pillow, NeurIPS 2020.

Link to paper: [put link here]

* We want to predict neural responses of higher-order visual cortex (e.g., V4, IT) from natural images. 
* We use a deep neural network to make this prediction, which require a *lot* of data to train accurately---data we don't have in neuroscience. 
* We reduce the amount of training data required by using high-contrast, binarized *gaudy images*.
* Training on *gaudy images* yields better DNN prediction than training on the same number of normal images.
* In our paper, we find this is because gaudy images overemphasize edges in the image.

# gaudy transformation
To transform a normal, colorful image into its gaudy version:

<img src="/extra/gaudy_transformation.png" width="741" height="221">

In code:

```python
img = download_image()  # download example img with shape 224 x 224 x 3
mean_img = np.mean(img)  # take mean across all pixels and channels
img_gaudy = np.copy(img)
img_gaudy[img_gaudy < mean_img] = 0   # transform each pixel to the rails
img_gaudy[img_gaudy >= mean_img] = 255
plt.imshow(img)
plt.imshow(img_gaudy)
```


Code for this paper will be uploaded soon.
If you need it before then, please contact the first author at bcowley at princeton edu.
