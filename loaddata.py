#Here we are importing numpy for all our numerical operations along with dataset loading
#We are going to use Keras based sequential model with Dense layers for our initial ANN
import numpy as np
import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy
from numpy import loadtxt
import keras
import keras.models
import keras.layers
from keras.models import Sequential
from keras.layers import Dense
df = pd.read_csv('/Users/dponnappan/Downloads/sales_0_0_0.csv').fillna(0)
col_list = list(df)
col_list[5], col_list[9] =col_list[9], col_list[5]
df.columns = col_list
prof = ProfileReport(df)
prof.to_file(output_file='lulusales_eda.html')

dataset = df.values

X = dataset[:,0:9]
X = np.asarray(X).astype('float32')
y = dataset[:,9]
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=150,batch_size=10)
_, accuracy = model.evaluate(X,y)
print("Accuracy %2f" %(accuracy*100))

