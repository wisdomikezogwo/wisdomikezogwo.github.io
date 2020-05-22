---
layout: post
title:  "Transforming EEG signals into 'Images'"
date:   2019-10-23 00:38:00
categories: Research
---


This article discusses the various methods developed by researchers to transform physiological signals
e.g EEG (Electroencephalograms) to 2D 'image' representation. These developments have been driven by the
successes of `convolutional neural networks` in computer vision, an extensively studied deep learning
method for learning the contents of images and accurately classifying unseen images.

Over the years various methods of transforming this multivariate time series
signals measured from the brain, into 2 or 3-dimensional image-like tensors, have been proposed. This short article serves only to enlighten
the reader about a few of these techniques, by discussing the disparate functions that work together to make these methods work parts, whilst providing basic code for the reader to test these methods.

`Please Note` The language and 'vocab' used in this post has been
reduced to accommodate anyone interested in implementing these methods on there own without prior knowledge of the field or its nomenclature.

 The most common methods include:
 1. Rasterization            &emsp; &emsp;   &emsp; &emsp;  &emsp; &emsp;   *[Bashivan et. al (2016)][Bashivan]*
 2. Recurrence plots       &emsp; &emsp;    &emsp; &emsp;    &emsp;  *[Hatami et al (2017)][Hatami]*
 3. Wavelet time-freq. plots    &emsp; &emsp;      *[B. Xu et al. (2019)][B. Xu]*

 In this article, however, I would only be discussing the first two methods as they
 have proven to perform better, giving their utilization in the field.

# Rasterization
This image-based representation process first proposed by *[bashivan et. al (2016)][Bashivan]* in their paper *"Learning Representations from EEG With Deep
Recurrent-Convolutional Neural Networks"*  preserve the structure of the EEG data within space (spatial), frequency (spectral) and time
(temporal), by transforming the data into a multi-dimensional tensor. The method which sequentially
follows the steps below have found a great audience, one of which is the *'Learning Robust Features using Deep Learning for
Automatic Seizure Detection'* by *Thodoroff et al (2016)*.
* “ Transform the measurements into a 2D image. ” - preserves the spatial structure.
* “ Use multiple color channel representing the different frequency bands. ” - gives structure to the information across bands.
* “ Use a sequence of the images derived from consecutive time windows. ” - preserving
temporal evolution.

To achieve the first two points above the following methods where used:

(1).   `Azimuthal Equidistant Projection` (polar projection) is used to project the 3D coordinates of the 10-
20-system EEG data input to 2D, while preserving the relative distance between neighboring
electrodes.

{% highlight ruby %}
#modified from https://github.com/pbashivan/EEGLearn
def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)
{% endhighlight %}

![Rasters](/assets/3d_electrode_plot.png){:class="img-responsive"}{:height="250px" width="400px"}
![Rasters](/assets/2d_electrode_plot.png){:class="img-responsive"}{:height="250px" width="400px"}

(2).   `Clough-Tocher Scheme`: is used for
* Interpolating the scattered power measurements, and
* Estimating the values in-between electrodes over a 32 X 32 mesh.
This process is done repeatedly over different frequency (theta[4-7Hz], alpha[8-13Hz], beta[13-
30Hz]).

{% highlight ruby %}
# Interpolate the values
grid_x, grid_y = np.mgrid[
                 min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                 min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                 ]
temp_interp = []
for c in range(n_colors):
    temp_interp.append(np.zeros([n_samples, n_gridpoints, n_gridpoints]))

# Generate edgeless images
if edgeless:
    min_x, min_y = np.min(locs, axis=0)
    max_x, max_y = np.max(locs, axis=0)
    locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
    for c in range(n_colors):
        feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((n_samples, 4)), axis=1)

# Interpolating
for i in range(n_samples):
    for c in range(n_colors):
        temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                           method='cubic', fill_value=np.nan)
    print('Interpolating {0}/{1}\r'.format(i + 1, n_samples), end='\r')
{% endhighlight %}

The three maps are then merged to form one image with 3 (color) channels
to give a tensor with dimensions (samples, colors, W, H) of which random samples are displayed
below.
{% highlight ruby %}
images = gen_images(np.array(locs_2d), av_feats, 32, normalize=True)
# sample = 200
image = np.transpose(images[sample, :, :, :], (2, 1, 0))
plt.imshow(image)
{% endhighlight %}


![Rasters](/assets/BITMAP2000.png){:class="img-responsive"}{:height="200px" width="200px"}
![Rasters](/assets/BITMAP200.png){:class="img-responsive"}{:height="200px" width="200px"}
![Rasters](/assets/BITMAP9.png){:class="img-responsive"}{:height="200px" width="200px"}
![Rasters](/assets/BITMAP101.png){:class="img-responsive"}{:height="200px" width="200px"}

&nbsp;
&nbsp;
&emsp;
&ensp;
# Texture Images
Texture images are a time-series image-based representation produced by Recurrence plots (RP).
RP is a visualization tool that aims to explore the m-dimensional phase space
the trajectory through a 2D representation of its recurrences.
Unlike the Rasterization method which takes in a multivariate set of time series signals
this method takes in univariate time series signals.

The R-matrix produced plotted contains *texture* like single dots, diagonal lines as well as vertical and
horizontal lines; and typology information which is characterized as homogeneous, periodic, drift and disrupted.
> For instance, fading to the upper left and lower right corners means that the process contains a trend or drift;
or vertical and horizontal lines/clusters show that some states do not change or change slowly for some time and
this can be interpreted as laminar states ------- *[Hatami et al (2017)][Hatami]*


{% highlight ruby %}
import pylab as plt
from scipy.spatial.distance import pdist, squareform

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def moving_average(s, r=5):
    return np.convolve(s, np.ones((r,))/r, mode='valid')
{% endhighlight %}

{% highlight ruby %}
eps = 0.1
steps = 10
ru = sample_chb                 #Pre Extracted times series from CHB-MIT dataset
ru_filtered = moving_average(ru)

plt.title("Normal")
plt.subplot(221)
plt.plot(ru_filtered)
plt.title("EEG One Channel")
plt.subplot(223)
plt.imshow(rec_plot(ru_filtered, eps=eps, steps=steps))

rn = another_sample_chb         #Pre Extracted times series from CHB-MIT dataset
rn_filtered = moving_average(rn)

plt.subplot(222)
plt.plot(rn_filtered)
plt.title("Another channel")
plt.subplot(224)
plt.imshow(rec_plot(rn_filtered, eps=eps, steps=steps))

plt.show()
{% endhighlight %}

For different channels (electrode-pair) we could get the following:

![Textures](/assets/k2.png){:class="img-responsive"}{:height="250px" width="430px"}
![Textures](/assets/k3.png){:class="img-responsive"}{:height="250px" width="430px"}

This [github repo][gitu] contains the complete code and examples for these techniques discussed.

Br.

[Bashivan]:  https://arxiv.org/abs/1511.06448
[Hatami]:   https://arxiv.org/abs/1710.00886
[B. Xu]:   http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8585027&isnumber=8600701
[gitu]:    https://github.com/wisdomikezogwo/Timeseries-to-image-Transformer
