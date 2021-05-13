import matplotlib.pyplot as plt
import numpy as np

def show_images(image_dict, fileout="images.png", cols = 10, rows=10):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    '''
    n_images = len(images)
    #if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image) in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, interpolation='nearest', cmap='gray')
       # a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    '''

    fig, ax = plt.subplots(rows, cols)
    np.vectorize(lambda ax:ax.axis('off'))(ax)
    #fig.patch.set_visible(False)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    #fig = plt.figure()
    for key, images in image_dict.items():
        n_images = len(images)
        for n, image in enumerate(images):
            ax[key,n].imshow(image)
            if n == 0:
                ax[key,n].text(-28//2,28//2, key)
            #ax[key,n].axis('off')
        #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.savefig(fileout)
