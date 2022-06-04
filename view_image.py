import matplotlib.pyplot as plt
import numpy as np
import h5py 

slice = 10

file = h5py.File('localData.h5','r')

img = file['data'][slice,]
print(file['labels'][slice,])

labelsList = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']

img[:,:,0] = img[:,:,0]/np.max(img[:,:,0])*255
img[:,:,1] = img[:,:,1]/np.max(img[:,:,1])*255
img[:,:,2] = img[:,:,2]/np.max(img[:,:,2])*255

plt.imshow(np.uint8(img,cmap='gray'))
plt.axis('off')
plt.title('Monoclinic')
# plt.imshow(np.uint8(img/np.max(img)*255))
plt.show()
