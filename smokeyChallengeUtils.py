from scipy.fftpack import fft2, fftshift 
from scipy import ndimage
from tqdm import tqdm
import numpy as np
import h5py 

def ZoomInterpol(FFT_image, zoom_factor=1, interpol_factor=1):
    """Given a  2D FFT , return a 2D FFT with a zoom factor and interpolation if desired
    Parameters
    ----------
    FFT_image : 2D numpy array
        Numpy matrix of size (m,m). 2D FFT.
    zoom_factor: integer 
        (optional, default=1). Factor by which to magnify the FFT.
    interpol_factor: integer (optional, default = 1). Factor by which to increase the size of the image by interpolation.
        Array of size (nx2), where n is the number of classes as defined in the class dictionary

    Returns
    -------

    2D Numpy array 
        Numpy array that has been zoomed and inteprolated by the requested amounts.

    """

    zoom_size = (FFT_image.shape[0] / zoom_factor) / 2
    window_size_x = FFT_image.shape[0]
    window_size_y = FFT_image.shape[1]

    if np.mod(FFT_image.shape[0] / zoom_factor, 2) == 0:
        F2_zoomed = FFT_image[int(window_size_x / 2 - zoom_size):int(window_size_x / 2 + zoom_size),
                    int(window_size_y / 2 - zoom_size):int(window_size_y / 2 + zoom_size)]
    else:
        F2_zoomed = FFT_image[int(window_size_x / 2 - zoom_size):int(window_size_x / 2 + 1 + zoom_size),
                    int(window_size_y / 2 - zoom_size):int(window_size_y / 2 + 1 + zoom_size)]

    return ndimage.zoom(F2_zoomed, interpol_factor)

def extractTrainingDataset(fname):
  imgsFFT = np.zeros([3,360,360], dtype=np.complex64)

  file = h5py.File(fname, mode='r')

  ind = 0
  xTotal = np.zeros([360, 360,1])
  for batch in tqdm(file):
    for sample in file[batch]:
      #print(batch) 
      if batch != 'point_group_labels':
        imgs = file[batch][sample]['cbed_stack']
        
        for ii in range(imgs.shape[0]):
          imgsFFT[ii,] = fftshift(fft2(imgs[ii,75:435,75:435]))
        
        fftImg0 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[0,]), zoom_factor=4, interpol_factor=4))+1)
        if ind == 0:
          xTotal[:,:,ind] = fftImg0
          ind += 1
        else:
          xTotal = np.dstack((xTotal,fftImg0))

        fftImg1 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[1,]), zoom_factor=4, interpol_factor=4))+1)
        xTotal = np.dstack((xTotal,fftImg1))

        fftImg2 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[2,]), zoom_factor=4, interpol_factor=4))+1)
        xTotal = np.dstack((xTotal,fftImg2))

  xTotal = np.swapaxes(xTotal, 0, 2)

  return xTotal


def extractPartialTrainingDataset(fname, file):

  imgsFFT = np.zeros([3,360,360], dtype=np.complex64)

  ind = 0
  xTotalGray = np.zeros([360, 360, 3], dtype=np.float32)
  xTotalRGB  = np.zeros([360, 360, 3], dtype=np.float32)

  for sample in file:
    imgs = file[sample]['cbed_stack']

    for ii in range(imgs.shape[0]):
      imgsFFT[ii,] = fftshift(fft2(imgs[ii,75:435,75:435]))
      
    fftImg0 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[0,]), zoom_factor=4, interpol_factor=4))+1)
    fftImg1 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[1,]), zoom_factor=4, interpol_factor=4))+1)
    fftImg2 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[2,]), zoom_factor=4, interpol_factor=4))+1)
      
    if ind == 0:
      xTotalGray = np.swapaxes(np.array([fftImg0, fftImg0, fftImg0]), 0, 2).reshape(1,360,360,3)
      xTotalRGB  = np.swapaxes(np.array([fftImg0, fftImg1, fftImg2]), 0, 2).reshape(1,360,360,3)
      ind += 1
    else:
      xTotalGray = np.concatenate((xTotalGray, np.swapaxes(np.array([fftImg0, fftImg0, fftImg0]), 0, 2).reshape(1,360,360,3)), axis=0)
      xTotalRGB  = np.concatenate((xTotalRGB,  np.swapaxes(np.array([fftImg0, fftImg1, fftImg2]), 0, 2).reshape(1,360,360,3)), axis=0)
      
    xTotalGray = np.concatenate((xTotalGray, np.swapaxes(np.array([fftImg1, fftImg1, fftImg1]), 0, 2).reshape(1,360,360,3)), axis=0)
    xTotalGray = np.concatenate((xTotalGray, np.swapaxes(np.array([fftImg2, fftImg2, fftImg2]), 0, 2).reshape(1,360,360,3)), axis=0)
    
  return (xTotalGray, xTotalRGB)

def extractPartialGrayDataset(fname, file):

  imgsFFT = np.zeros([3,360,360], dtype=np.complex64)

  ind = 0
  xTotalGray = np.zeros([360, 360, 3], dtype=np.float32)
  xTotalRGB  = np.zeros([360, 360, 3], dtype=np.float32)

  for sample in file:
    imgs = file[sample]['cbed_stack']

    for ii in range(imgs.shape[0]):
      imgsFFT[ii,] = fftshift(fft2(imgs[ii,75:435,75:435]**0.2))

    fftImg0 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[0,]), zoom_factor=4, interpol_factor=4))+1)
    fftImg1 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[1,]), zoom_factor=4, interpol_factor=4))+1)
    fftImg2 = np.log(np.abs(ZoomInterpol(np.abs(imgsFFT[2,]), zoom_factor=4, interpol_factor=4))+1)

    fftImg0 /= np.amax(fftImg0)
    fftImg1 /= np.amax(fftImg1)
    fftImg2 /= np.amax(fftImg2)

    if ind == 0:
      xTotalGray = np.swapaxes(np.array([fftImg0, fftImg0, fftImg0]), 0, 2).reshape(1,360,360,3)
      ind += 1
    else:
      xTotalGray = np.concatenate((xTotalGray, np.swapaxes(np.array([fftImg0, fftImg0, fftImg0]), 0, 2).reshape(1,360,360,3)), axis=0)

    xTotalGray = np.concatenate((xTotalGray, np.swapaxes(np.array([fftImg1, fftImg1, fftImg1]), 0, 2).reshape(1,360,360,3)), axis=0)
    xTotalGray = np.concatenate((xTotalGray, np.swapaxes(np.array([fftImg2, fftImg2, fftImg2]), 0, 2).reshape(1,360,360,3)), axis=0)

  return xTotalGray

def create_generator(fname):
  from hdf5_preprocessing import HDF5ImageGenerator

  h5File = h5py.File(fname,'r')
  nSamples = h5File['data'].shape[0]
  dataFormat = 'channels_last'

  dataGen = HDF5ImageGenerator(data_format=dataFormat)
  generator = dataGen.flow_from_hdf5(h5File,batch_size = 1, 
        shuffle = False,
        offset=0, nsample=nSamples)

  return generator

def train_data_parallel_generator(fname, batchSize, hvd):
  from hdf5_preprocessing import HDF5ImageGenerator

  h5File = h5py.File(fname,'r')

  nSamples = h5File['data'].shape[0]
  nSamples_loc = nSamples // hvd.size()
  train_offset = nSamples_loc * hvd.rank()

  if hvd.rank() == 0:
    print(fname + ', batchSize '+str(batchSize)+', nSamples: ' + str(nSamples) + ', nSamples_loc: ' + str(nSamples_loc))

  dataFormat = 'channels_last'

  dataGen = HDF5ImageGenerator(
	horizontal_flip = True,
        vertical_flip = True,
        fill_mode = "reflect",
        #shear_range = 0.2,        
        zoom_range = 0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        #rotation_range=15,
        data_format=dataFormat)

  generator = dataGen.flow_from_hdf5(h5File,batch_size = batchSize,
        shuffle = False,
        offset=train_offset, nsample=nSamples_loc)

  return generator

def valid_data_parallel_generator(fname, hvd):
  from hdf5_preprocessing import HDF5ImageGenerator

  h5File = h5py.File(fname,'r')

  nSamples = h5File['data'].shape[0]
  nSamples_loc = nSamples // hvd.size()
  valid_offset = nSamples_loc * hvd.rank()

  dataFormat = 'channels_last'
  dataGen = HDF5ImageGenerator(data_format=dataFormat)

  generator = dataGen.flow_from_hdf5(h5File,batch_size = 1,
        shuffle = False,
        offset=valid_offset, nsample=nSamples_loc)

  return generator


def get_model(nClasses):
  from keras.layers import Dense, Conv2D, Dropout, AveragePooling2D, Flatten
  from keras.models import Sequential
  from keras.regularizers import l2

  #create model
  model = Sequential()

  #add model layers
  model.add(Conv2D(32,(5,5),activation='relu',kernel_regularizer=l2(0.005),input_shape=(360,360,1)))
  model.add(Conv2D(32,(5,5),activation='relu',kernel_regularizer=l2(0.005)))
  model.add(AveragePooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(64,(5,5),activation='relu',kernel_regularizer=l2(0.005)))
  model.add(Conv2D(64,(5,5),activation='relu',kernel_regularizer=l2(0.005)))
  model.add(AveragePooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128,(5,5),activation='relu',kernel_regularizer=l2(0.005)))
  model.add(AveragePooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(nClasses, activation='softmax'))

  return model

def get_dl_model(nClasses, model):
  from tensorflow.keras.layers import Dense, Dropout, AveragePooling2D, GlobalMaxPooling2D,  Flatten
  import tensorflow.keras.applications as app
  from tensorflow.keras.models import Model 

  if model == 'resnet50':
    baseModel = app.resnet50.ResNet50(include_top=False, weights=None, input_shape=(360,360,3))
  elif model == 'resnet101':
    baseModel = app.resnet.ResNet101(include_top=False, weights=None, input_shape=(360,360,3))
  elif model == 'xception':
    baseModel = app.xception.Xception(include_top=False, weights=None, input_shape=(360,360,3))
  elif model == 'inception':
    baseModel = app.inception_v3.InceptionV3(include_top=False, weights=None, input_shape=(360,360,3))
  else:
    print('Incorrect Model is Passed!!!')

  headModel = baseModel.output

  headModel = AveragePooling2D(pool_size=(7,7))(headModel)
  headModel = Flatten()(headModel)
  
  headModel = Dense(256, activation='relu')(headModel)
  headModel = Dropout(0.4)(headModel)
  #headModel = Dense(128, activation='relu')(headModel)
  #headModel = Dropout(0.2)(headModel)
  headModel = Dense(nClasses, activation='softmax')(headModel)

  model = Model(inputs=baseModel.input, outputs=headModel)

  return model

def weighted_loss(weights):
  # Weighted loss from Yin, Cui & Menglin, Jia et. al. 
  # (Class Balanced Loss on Effective Number of Samples)
  import tensorflow as tf

  def weighted_cross_entroy(y_true, y_pred):

    weights = tf.cast(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights,0)
    weights = tf.tile(weights [tf.shape(y_pred)[0], 1]) * y_pred
    weights = tf.reduce_sum(weights, axis = 1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, num_classes])

    return weights * tf.keras.losses.categorical_crossentropy(y_true, ypred)


