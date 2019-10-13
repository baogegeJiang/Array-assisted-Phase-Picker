from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, GRU, MaxPooling1D, AveragePooling1D, SimpleRNN,Softmax,Activation,Dropout
from keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D,MaxPooling2D,Reshape
import math
import scipy.io as sio
import numpy as np
import os
matFile='G:\\data\\SCXR.mat'
matLoad = sio.loadmat(matFile)
dataX = matLoad['px']
dataY=matLoad['py']
model = Sequential()
in_features = 3
n_1 = 2000
n_2=1
n_out = 24
out_dim1 = 100
out_dim2 = 50
out_features1 = 100
out_features2 = 100
filt1=32
filt2=32
filt3=48
filt4=64
filt5=128
filt6=128
filt7=128
filterT1=50
filterT2=10
filterT3=4
filterT4=1

model.add(Conv2D(filt1,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu',input_shape=(n_1,n_2, in_features)))
model.add(MaxPooling2D(pool_size=(4,1), strides=(4,1),padding='same'))
model.add(Conv2D(filt2,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(4,1), strides=(4,1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filt3,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='same'))
model.add(Conv2D(filt4,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='same'))
model.add(Conv2D(filt5,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(4,1), padding='same'))
model.add(Conv2D(filt6,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='same'))
model.add(Conv2D(filt7,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(4,1), padding='same'))
#model.add(Reshape((-1,)))
#model.add(Dense(out_features1,activation='relu'))
#model.add(Dense(out_features2,activation='relu'))
#model.add(Reshape((10,1,10)))
model.add(Conv2DTranspose(filterT1,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu'))
model.add(Conv2DTranspose(filterT2,kernel_size=(20,1),strides=(10,1),padding='same',activation='tanh'))
model.add(Conv2DTranspose(filterT3,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu'))
model.add(Conv2DTranspose(filterT4,kernel_size=(20,1),strides=(2,1),padding='same',activation='tanh'))
model.add(activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
print(model.summary())
resFile='G:\\data\\resData.mat'

tmp0=0
count=0
for i in range(30):
    index=math.floor(np.random.rand()*5000)
    ne=20
    if i>10 :    
        ne=5
    model.fit(dataX[(index+1000):(index+3000),:,:,:],dataY[(index+1000):(index+3000),:,:,0].reshape([-1,2000,1,1]), nb_epoch=ne, batch_size=500, verbose=2)
    #tmp=model.predict( dataX[100:300,:,:,0:3])
    #maxIndex=np.array(tmp.argmax(axis=1))
    #maxIndexO=np.array(dataY[100:300,:,:,0]).argmax(axis=1)
    #print((maxIndex-maxIndexO).mean(),(maxIndex-maxIndexO).var())
    #print(tmp[2])
   # if (tmp[2]<tmp0) & (i>1):
  #      count=count+1
 #   else:
 #       tmp0 = tmp[2
   #      count=0
  #  if count>3:
  #      break
outY = model.predict(dataX[0:100,:,:,:], verbose=0)
sio.savemat(resFile,{'outy':outY})
#print(model.summary())
filepath ='G:\\data\\model'
model.save(filepath)
#keras.models.load_model(filepath)

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, GRU, MaxPooling1D, AveragePooling1D, SimpleRNN,Softmax,Activation,Dropout
from keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D,MaxPooling2D,Reshape
import math
import scipy.io as sio
import numpy as np
import os
matFile='G:\\data\\SCXR.mat'
matLoad = sio.loadmat(matFile)
dataX = matLoad['sx']
dataY=matLoad['sy']
model = Sequential()
in_features = 3
n_1 = 2000
n_2=1
n_out = 24
out_dim1 = 100
out_dim2 = 50
out_features1 = 100
out_features2 = 100
filt1=32
filt2=32
filt3=48
filt4=64
filt5=128
filt6=128
filt7=128
filterT1=50
filterT2=10
filterT3=4
filterT4=1

model.add(Conv2D(filt1,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu',input_shape=(n_1,n_2, in_features)))
model.add(MaxPooling2D(pool_size=(4,1), strides=(4,1),padding='same'))
model.add(Conv2D(filt2,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(4,1), strides=(4,1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filt3,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='same'))
model.add(Conv2D(filt4,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='same'))
model.add(Conv2D(filt5,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(4,1), padding='same'))
model.add(Conv2D(filt6,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1), padding='same'))
model.add(Conv2D(filt7,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,1), strides=(4,1), padding='same'))
#model.add(Reshape((-1,)))
#model.add(Dense(out_features1,activation='relu'))
#model.add(Dense(out_features2,activation='relu'))
#model.add(Reshape((10,1,10)))
model.add(Conv2DTranspose(filterT1,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu'))
model.add(Conv2DTranspose(filterT2,kernel_size=(20,1),strides=(10,1),padding='same',activation='tanh'))
model.add(Conv2DTranspose(filterT3,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu'))
model.add(Conv2DTranspose(filterT4,kernel_size=(20,1),strides=(2,1),padding='same',activation='tanh'))
model.add(activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
print(model.summary())
resFile='G:\\data\\resData.mat'

tmp0=0
count=0
for i in range(30):
    index=math.floor(np.random.rand()*5000)
    ne=20
    if i>10 :    
        ne=5
    model.fit(dataX[(index+1000):(index+2000),:,:,:],dataY[(index+1000):(index+2000),:,:,0].reshape([-1,2000,1,1]), nb_epoch=ne, batch_size=500, verbose=2)
    #tmp=model.predict( dataX[100:300,:,:,0:3])
    #maxIndex=np.array(tmp.argmax(axis=1))
    #maxIndexO=np.array(dataY[100:300,:,:,0]).argmax(axis=1)
    #print((maxIndex-maxIndexO).mean(),(maxIndex-maxIndexO).var())
    #print(tmp[2])
   # if (tmp[2]<tmp0) & (i>1):
  #      count=count+1
 #   else:
 #       tmp0 = tmp[2
   #      count=0
  #  if count>3:
  #      break
outY = model.predict(dataX[0:100,:,:,:], verbose=0)
sio.savemat(resFile,{'outy':outY})
#print(model.summary())
filepath ='G:\\data\\modelS'
model.save(filepath)



from keras.models import *
from keras.layers import LSTM, Dense, TimeDistributed, GRU, MaxPooling1D, AveragePooling1D, SimpleRNN,Softmax,Activation,Dropout
from keras.layers import *
import math
import scipy.io as sio
import numpy as np
import os
matFile='G:\\data\\SCXR.mat'
matLoad = sio.loadmat(matFile)
dataX = matLoad['sx']
dataY=matLoad['sy']

in_features = 3
n_1 = 2000
n_2=1
n_out = 24
out_dim1 = 100
out_dim2 = 50
out_features1 = 100
out_features2 = 100
vectorLen=4;
filt1=32
filt2=32
filt3=48
filt4=64
filt5=128
filt6=128
filt7=128
filterT5=50
filterT4=10
filterT3=4
filterT2=1

inputs=Input((n_1,n_2, in_features));
conv1=Conv2D(filt1,kernel_size=(10,1) ,strides=(1,1), padding='same',activation='relu')(inputs)
pool1=MaxPooling2D(pool_size=(10,1), strides=(5,1),padding='same')(conv1)

conv2=Conv2D(filt2,kernel_size=(10,1) ,strides=(1,1), padding='same',activation='relu')(pool1)
pool2=AveragePooling2D(pool_size=(10,1), strides=(5,1), padding='same')(conv2)
pool2=Dropout(0.2)(pool2)

conv3=Conv2D(filt3,kernel_size=(10,1) ,strides=(1,1), padding='same',activation='tanh')(pool2)
pool3=MaxPooling2D(pool_size=(10,1), strides=(5,1),padding='same')(conv3)

conv4=Conv2D(filt4,kernel_size=(8,1) ,strides=(1,1), padding='same',activation='relu')(pool3)
pool4=AveragePooling2D(pool_size=(8,1), strides=(4,1), padding='same')(conv4)
pool4=Dropout(0.2)(pool4)


conv5=Conv2D(filt5,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='tanh')(pool4)
pool5=MaxPooling2D(pool_size=(4,1), strides=(2,1),padding='same')(conv5)

conv6=Conv2D(filt6,kernel_size=(4,1) ,strides=(1,1), padding='same',activation='relu')(pool4)
pool6=MaxPooling2D(pool_size=(4,1), strides=(2,1),padding='same')(conv6)

vector=Reshape([-1])(pool6)
vector=Dense(vectorLen*filt6,activation='relu')(vector)

dConv6=Reshape([-1,1,filt6])(vector)
dConv6=concatenate([dConv6,conv6],axis=3)

dConv5=Conv2DTranspose(filterT5,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu')(dConv6)
dConv5=concatenate([dConv5,conv5],axis=3)

dConv4=Conv2DTranspose(filterT4,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu')(dConv5)
dConv4=dConv4.concatenate(conv4,axis=3)

dConv3=Conv2DTranspose(filterT3,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu')(dConv4)
dConv3=dConv3.concatenate(conv3,axis=3)

dConv2=Conv2DTranspose(filterT2,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu')(dConv3)
dConv2=dConv2.concatenate(conv2,axis=3)

dConv1=Conv2DTranspose(filterT1,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu')(dConv2)
dConv1=dConv1.concatenate(conv1,axis=3)

output=onv2DTranspose(in_features,kernel_size=(20,1),strides=(10,1),padding='same',activation='relu')(dConv1)

output=output.add(activation='sigmoid')
model=Model(input=inputs,output=output)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
print(model.summary())
resFile='G:\\data\\resData.mat'

tmp0=0
count=0
for i in range(30):
    index=math.floor(np.random.rand()*5000)
    ne=20
    if i>10 :    
        ne=5
    model.fit(dataX[(index+1000):(index+2000),:,:,:],dataY[(index+1000):(index+2000),:,:,0].reshape([-1,2000,1,1]), nb_epoch=ne, batch_size=500, verbose=2)
    #tmp=model.predict( dataX[100:300,:,:,0:3])
    #maxIndex=np.array(tmp.argmax(axis=1))
    #maxIndexO=np.array(dataY[100:300,:,:,0]).argmax(axis=1)
    #print((maxIndex-maxIndexO).mean(),(maxIndex-maxIndexO).var())
    #print(tmp[2])
   # if (tmp[2]<tmp0) & (i>1):
  #      count=count+1
 #   else:
 #       tmp0 = tmp[2
   #      count=0
  #  if count>3:
  #      break
outY = model.predict(dataX[0:100, :, : , : ], verbose=0)
sio.savemat(resFile, {'outy':outY})
#print(model.summary())
filepath = 'G:\\data\\modelS'
model.save(filepath )