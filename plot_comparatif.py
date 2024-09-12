import numpy as np
import matplotlib.pyplot as plt
import jax 
import cv2
import colorsys
import matplotlib
import matplotlib.colors as mcolors

def man_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:,:3]])
    hls[:,1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)

x = 31

cv_att = cv2.flip(cv2.rotate(np.load('array_final copy/chan_vese_epoch-700_idx-'+str(x)+'.npy'), cv2.ROTATE_90_COUNTERCLOCKWISE),1)

att = cv2.flip(cv2.flip(cv2.rotate(np.load('array_final/chan_vese_epoch-900_idx-'+str(x)+'.npy'),cv2.ROTATE_90_COUNTERCLOCKWISE),1),0)
img = jax.image.resize(cv2.rotate(np.load('array_final copy/img.npy'),cv2.ROTATE_90_CLOCKWISE),shape=(512,512, 3) ,method= 'cubic', antialias=True)

img = cv2.cvtColor(np.array(jax.numpy.clip(img * jax.numpy.array([0.229, 0.224, 0.225]) + jax.numpy.array([0.485, 0.456, 0.406]), 0,1)),cv2.COLOR_BGR2GRAY)

mask = cv2.rotate(np.load('mask.npy'),cv2.ROTATE_90_CLOCKWISE)
cmap = plt.cm.get_cmap("rainbow")

l1 = [cv_att, att]

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(29, 10))
plt.subplots_adjust(wspace=0, hspace=0)
axes[0].imshow(img, cmap='gray')

for spine in ["left", "right", "top", "bottom"]:
    axes[0].spines[spine].set_visible(False)

axes[0].set_xticks([])
axes[0].set_yticks([])

axes[0].set_aspect('equal')
axes[1].imshow(mask, cmap='gray')

for spine in ["left", "right", "top", "bottom"]:
    axes[1].spines[spine].set_visible(False)

axes[1].set_xticks([])
axes[1].set_yticks([])

axes[1].set_aspect('equal')


for idx,i in enumerate(l1):
    
    axes[idx+2].imshow(img, cmap='gray')

    
    contour1 = jax.image.resize(i.squeeze(), img.shape[:2], 'cubic', True)


    if idx==1:
        im = axes[idx+2].imshow(contour1, cmap=man_cmap(cmap = plt.cm.get_cmap("rainbow_r"), value=.75), alpha=0.4)
    else:
        im = axes[idx+2].imshow(contour1, cmap=man_cmap(cmap = plt.cm.get_cmap("rainbow"), value=.75), alpha=0.4)
    for spine in ["left", "right", "top", "bottom"]:
        axes[idx+2].spines[spine].set_visible(False)

    axes[idx+2].set_xticks([])
    axes[idx+2].set_yticks([])

    axes[idx+1].set_aspect('equal')



plt.savefig('comparison_plot.png', bbox_inches='tight')