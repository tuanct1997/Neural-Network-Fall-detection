import numpy as np
from numpy import zeros, newaxis
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Reshape, SimpleRNN
from sklearn.model_selection import StratifiedKFold
# from keras.optimizers import SGD
# from keras.activations import elu
from keras import initializers, losses
from keras.callbacks import ModelCheckpoint,History
from keras.models import load_model
import matplotlib.pyplot as plt
epochs = 200
batch_size = 100
seed = 5
np.random.seed(seed)

dataset = np.loadtxt("newData.txt",delimiter=";")
# testset = np.loadtxt("testData.txt",delimiter=";")

X = dataset[:,0:9]
Y = dataset[:,9:10]
# y_trains = dataset[:,9:10]
# y_test = testset[:,9:10]

# print np.shape(x_trains)
# print np.shape(x_test)
# print np.shape(y_trains)
# print np.shape(y_test)

# x_trains.reshape(550,10,9)
# x_test.reshape(41,10,9)
# y_trains.reshape(550,10,1)
# y_test.reshape(41,10,1)
kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(X, Y):
	model = Sequential()
# # LSTM(units=128, input_shape=X_train.shape[1:]))

	model.add(Dense(16, activation='relu', input_dim=9))
	model.add(Dropout(0.2))
# # model.add(Reshape((64,1)))
# # model.add(LSTM(64,activation = 'relu',return_sequences=False))#True = many to many
# # model.add(Dropout(0.9))
# # model.add(SimpleRNN(64,activation = "relu",use_bias = False))
	model.add(Dense(8, activation='relu'))
	model.add(Dropout(0.2))
# # model.add(Dense(32, activation='relu')) # a brief experiment with 3 hidden layers...
# # model.add(Dropout(0.2))                    # ...that didn't go well.
# # model.add(Dense(16, activation='relu'))
# # model.add(Dropout(0.9))
	model.add(Dense(1, activation='relu'))
	model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))
# model.add(Dropout(0.2))
	initializers.TruncatedNormal(mean=1.0, stddev=0.05, seed=None)
	# model.summary()
	model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# with open('blank.txt', 'w') as f:
#     for item in callbacks_list:
#         f.write(str(item) + "\n")

# print "Training the model..."
	history = model.fit(X[train], Y[train],
          	batch_size=batch_size,
          	epochs=epochs,
          	verbose=0
          # validation_split = 0.25,
          # validation_data=(x_test, y_test),
          # shuffle=True)
          	)
	scores = model.evaluate(X[test], Y[test], verbose=0)


# print"Test loss:", score[0]
# print"Test accuracy:", score[1]
# print "abc"
	test = model.evaluate(X[test],Y[test],verbose = 0)
	print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
	cvscores.append(scores[1] * 100)
print "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))
print np.mean(cvscores)

# print test
# print "axz"
# test1 = model.predict(x = x_test,verbose = 1)
# print test1
# model.save('mymodel.h5')
# del model 
# history = load_model('mymodel.h5')
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
# best result : 2 linears, 1 relu, epochs = 33, batch = 128 , acc = 80.39%	