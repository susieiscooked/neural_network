#imshow function to create an image
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img/ 2+0.5 #to unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig('test_images.png')
    plt.show()
