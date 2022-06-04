from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import horovod.keras as hvd
import horovod.tensorflow.keras as hvd
import keras

import smokeyChallengeUtils as utils
import numpy as np
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-num_epoch', default=100,type=int, help='Number of Epochs')
parser.add_argument('-batch_size', default=32,type=int, help='Batch Size')
parser.add_argument('-run_name', default='v3',type=str, help='Run Name')
parser.add_argument('-color', default='RGB',type=str,help='RGB or Grayscale Training Data')
parser.add_argument('-learning_rate',default=0.0001, type=float, help='Base Learning Rate')
parser.add_argument('-optimizer', default='Adam', type=str, help='Optimizer (Adam or SGD)')
parser.add_argument('-model', default='resnet50',type=str, help='Load DL Model, Options:{resnet50,resnet101,xception,inception}')
parser.add_argument('-resume_from_epoch', default=False, type=bool, help='Resume from Latest Checkpoint') 
input = parser.parse_args()

# Parse Inputs
nEpochs = input.num_epoch
batchSize = input.batch_size
baseLR = input.learning_rate

labelsList = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']
directory = '/gpfs/alpine/scratch/jtschw/lrn007/smokeyChallenge/'
outName = input.color + '_' + input.optimizer + '_nEpochs_' + str(nEpochs) + '_batchSize_' + str(batchSize) + '_' + input.run_name

# Horovod: initialize Horovod. 
hvd.init()

# Print Command Line Inputs
if hvd.rank() == 0:
	print(' ==== Command Line Inputs ==== ')
	print(' DL Model: ' + input.model)
	print(' runName: ' + outName )
	print(' nEpochs: ' + str(nEpochs))
	print(' batchSize: ' + str(batchSize))
	print(' color: ' + input.color)
	print(' ============================= ')

# Horovod: set GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')	

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Restore from a Previous Checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights 
# to other workers.
if input.resume_from_epoch:
	import os
	for try_epoch in range(nEpochs, 0, -1):
		checkpointDir = directory+'results/'+input.model+'/'+input.color+'/'+outName+'/checkpoint-'+str(try_epoch)+'.h5'
		if os.path.exists(checkpointDir):
			input.resume_from_epoch = try_epoch
			break

	#input.resume_from_epoch = hvd.broadcast(input.resume_from_epoch, 0)

	if hvd.rank() == 0: model = hvd.load_model(checkpointDir)

# Generate New / Untrained  Model
else:
	model = utils.get_dl_model(len(labelsList), input.model)
	input.resume_from_epoch = 0

	# Determine Optimizer and Compile the Model
	if input.optimizer == 'Adam':
        	baseLR = 0.0001
        	opt = Adam(lr = baseLR * hvd.size())
	elif input.optimizer == 'SGD':
        	baseLR = 0.001
        	opt = SGD(lr = baseLR * hvd.size(), momentum = 0.9)
	else:
        	print('Incorrect Optimizer Selected, Exiting the Program')
        	exit()
	opt = hvd.DistributedOptimizer(opt)

	weights = (1 - beta ) / (1 - beta**nSamples)
	weights = weights / np.sum(weights) * len(nSamples)

	model.compile(optimizer=opt, loss=weighted_loss(weights), metrics=['accuracy'],experimental_run_tf_function=False)
	# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'],experimental_run_tf_function=False)

# Create Training and Validation Generator
train_generator = utils.train_data_parallel_generator(directory+'trainData/train'+input.color+'.h5',batchSize, hvd)
val_generator   = utils.valid_data_parallel_generator(directory+'trainData/validation'+input.color+'.h5', hvd)

callbacks = [
			# Horovod: broadcast initial variable states from rank 0 to all other processes.
			# This is necessary to ensure consistent initialization of all workers when
			# training is started with random weights or restored from a checkpoint.
			hvd.callbacks.BroadcastGlobalVariablesCallback(0),  

			# Horovod: average metrics among workers at the end of every epoch.
			# Note: This callback must be in the list before the ReduceLROnPlateau,
			# TensorBoard, or other metrics-based callbacks
			hvd.callbacks.MetricAverageCallback(), 
			
			# Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
			# accuracy. Scale the Learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during 
			# the first 10 epochs. See https://arxiv.org/abs/1706.02677 for details.
			hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose),  

			# Reduce the learning rate if training plateaues.
    		keras.callbacks.ReduceLROnPlateau(factor=0.25, verbose=verbose), ]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
	callbacks.append(keras.callbacks.ModelCheckpoint(directory + 'results/'+input.model+'/'+input.color+'/'+outName+'/checkpoint-{epoch}.h5'))
	callbacks.append(keras.callbacks.TensorBoard(directory + 'results/'+input.model+'/'+input.color+'/'+outName+'/'))

# Start Training
history = model.fit(	train_generator,
						steps_per_epoch = (train_generator.n ) // hvd.size(),
						epochs = nEpochs+input.resume_from_epoch, 
						initial_epoch = input.resume_from_epoch, 
						validation_data = val_generator, 
						validation_steps = 3*(val_generator.n ) // hvd.size(),
						verbose= verbose, callbacks = callbacks )

# Evaluate the Model on the Test Dataset
test_generator = utils.valid_data_parallel_generator(directory+'testData/test'+input.color+'.h5',hvd)
test_score = hvd.allreduce(model.evaluate(test_generator, verbose = verbose))

if hvd.rank() == 0:
	print(" Test %s: %.2f%%" % (model.metrics_names[1], test_score[1]*100))

	# Save Training History
	h5History = h5py.File(directory+'results/'+input.model+'/'+input.color+'/'+outName+'/bs_'+str(batchSize)+'_trainingHistory.h5', 'a')

	# Create datasets if 
	if input.resume_from_epoch == 0:
		for item in history.history:
			h5History.create_dataset(item, data=history.history[item],maxshape=(None,))
		h5History.create_dataset('test_final_score', data=test_score)
	else:
		for item in history.history:
			h5History[item][input.resume_from_epoch:input.resume_from_epoch+nEpochs] = history.history[item]
		h5History['test_final_score'][:] = test_score

	h5History.close()

	print("Saved model to disk")
