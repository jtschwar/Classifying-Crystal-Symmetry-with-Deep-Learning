from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py 

opt = 'Adam'
color = 'RGB'
nEpoch = 200
ii = 199

baseSaveDir = '/gpfs/alpine/scratch/jtschw/lrn007/smokeyChallenge/results/resnet50/'
saveDir = baseSaveDir+color+'/'+color+'_'+opt+'_nEpochs_'+str(nEpoch)+'_batchSize_'+str(bs)+'_v1/' 
cmFile = h5py.File(saveDir + 'confusion_matrices.h5','w') 
loadDir = saveDir + 'checkpoint-' + str(ii+1) + '.h5'
model = load_model(loadDir)

# Create Confusion Matrix for Training Data
print('Creating Confusion Matrix for Training Data')
loadDir = '/gpfs/alpine/scratch/jtschw/lrn007/smokeyChallenge/'
fname = loadDir +'trainData/train'+color+'.h5'
h5File = h5py.File(fname,'r')
train_images = h5File['data']
train_pred_raw = model.predict(train_images)
train_pred = np.argmax(train_pred_raw,axis=1)
train_labels = np.argmax(h5File['labels'],axis=1)
cm_train = confusion_matrix(train_labels, train_pred)
cmFile.create_dataset('cm_train',data = cm_train)
h5File.close()
#gc
train_pred_raw, train_pred, train_labels = None, None, None

print('Creating Confusion Matrix for Validation Data')
fname = loadDir +'trainData/validation'+color+'.h5'
h5File = h5py.File(fname,'r')
val_images = h5File['data']
val_pred_raw = model.predict(train_images)
val_pred = np.argmax(train_pred_raw,axis=1)
val_labels = np.argmax(h5File['labels'],axis=1)
cm_val = confusion_matrix(val_labels, val_pred)
cmFile.create_dataset('cm_val',data = cm_val)
h5File.close()
#gc
val_pred_raw, val_pred, val_labels = None, None, None

print('Creating Confusion Matrix for Test Data')
fname = loadDir +'testData/test'+color+'.h5'
h5File = h5py.File(fname,'r')
test_images = h5File['data']
test_pred_raw = model.predict(test_images)
test_pred = np.argmax(test_pred_raw,axis=1)
test_labels = np.argmax(h5File['labels'],axis=1)
cm_test = confusion_matrix(test_labels, test_pred)
cmFile.create_dataset('cm_test',data = cm_test)
h5File.close()
#gc
test_pred_raw, test_pred, test_labels = None, None, None
