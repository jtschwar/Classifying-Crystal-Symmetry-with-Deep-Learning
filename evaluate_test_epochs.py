from tensorflow.keras.models import load_model
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import smokeyChallengeUtils as utils
from tqdm import tqdm
import numpy as np
import h5py as h5

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-optimizer', default='Adam', type=str, help='Optimizer (Adam or SGD)')
parser.add_argument('-color', default='RGB',type=str,help='RGB or Grayscale Training Data')
parser.add_argument('-num_epoch', default=200,type=int, help='Number of Epochs')
input = parser.parse_args()

nEpoch = input.num_epoch
color = input.color
opt = input.optimizer
batchSize = [8, 16, 32]

loadDir = '/gpfs/alpine/scratch/jtschw/lrn007/smokeyChallenge/'
baseSaveDir = '/gpfs/alpine/scratch/jtschw/lrn007/smokeyChallenge/results/resnet50/'

# Horovod: initialize Horovod. 
hvd.init()

# Horovod: set GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')	

test_generator = utils.valid_data_parallel_generator(loadDir+'testData/test'+color+'.h5',hvd)
for bs in batchSize:

	saveDir = baseSaveDir+color+'/'+color+'_'+opt+'_nEpochs_'+str(nEpoch)+'_batchSize_'+str(bs)+'_v1/'	

	if hvd.rank() == 0:
		print('Processing Data for Batch Size: ' + str(bs))
		testEvals = np.zeros(nEpoch)
	
	for ii in tqdm(range(nEpoch)):
		loadDir = saveDir + 'checkpoint-' + str(ii+1) + '.h5'
		
		model = hvd.load_model(loadDir)

		testScore = hvd.allreduce(model.evaluate(test_generator, verbose = 0))	
		if hvd.rank() == 0:
			testEvals[ii] = testScore[1]

		tf.keras.backend.clear_session()
		del model

	if hvd.rank() == 0:
		outFile = h5.File(saveDir + '/' + 'bs_' + str(bs) + '_trainingHistory.h5', 'a')
		outFile.create_dataset('test_accuracy', data = testEvals)
		outFile.close()
 
