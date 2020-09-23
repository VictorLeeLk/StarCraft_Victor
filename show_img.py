import matplotlib.pyplot as plt
import numpy as np

from load_data import trainloader, classes
import torchvision
import sys
from signal import signal, SIG_DFL, SIG_IGN


def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

imgshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
