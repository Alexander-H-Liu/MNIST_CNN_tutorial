# Using keras, a well known deep learning framework.
# It's suitable for ppl who are new to deep learning (since it is a high level api)
import keras

# Import the components we're about to use
from keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense

# import mnist dataset using keras's api
from keras.datasets import mnist


# image shape of MNIST
image_shape = (28,28,1)

# Parameters for the model we're about to build
# Modify these values as you like!
layer_1_filters = 9
layer_1_kernel_size = 2
layer_2_filters = 9
layer_2_kernel_size = 2
nn_size = 32

# ------------------------------------------------------------------------ #
# # Dataset
# 
# MNIST : handwritten digit dataset.
print('-------------------------------Preprocess-------------------------------')

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the image
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Meta data of MNIST
print('\t\tMNSIT Dataset\t\t')
print('Training set size :\t',len(x_train))
print('Testing set size :\t',len(x_test))


# Pick a test image for demo
test_index = 0
test_image = x_test[test_index]
test_label = y_test[test_index]

# ------------------------------------------------------------------------ #
# # CNN Model
# Convolution Nerual Network for MNIST digit classification.
print('-------------------------------Model Construction-------------------------------')

# Using sequential model (strait forward pipeline with single input/output) from keras
model = keras.models.Sequential()

# Add convolution layers to our model
model.add(Conv2D(layer_1_filters,layer_1_kernel_size,input_shape=image_shape))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Conv2D(layer_2_filters,layer_2_kernel_size))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Flatten()) # reshape a n*n*depth image to a vector with dimension = n*n*depth
model.add(Dense(nn_size,activation='relu'))
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('\t\t\tOverview of Our CNN Model\t\t\t')
model.summary()


# ------------------------------------------------------------------------ #
# # Train model
# We train our CNN model with the training data from MNIST for 2 epoch
print('-------------------------------Training-------------------------------')

# Reshape input (specified image depth for keras cnn)
x_train = x_train.reshape(-1,28,28,1)
# one-hot encoding label for as the target of model
y_train = keras.utils.to_categorical(y_train, 10)

_ = model.fit(x = x_train, y = y_train, epochs = 2)


# ------------------------------------------------------------------------ #
# # Test model
# Verfy the performance of our CNN model with the testing data from MNIST.
print('-------------------------------Testing-------------------------------')

x_test = x_test.reshape(-1,28,28,1)
y_test = keras.utils.to_categorical(y_test, 10)
loss,acc = model.evaluate(x_test,y_test)
print('Testing loss =',loss)
print('Testing accuracy =',acc)

pred_label = model.predict(x_test[test_index:test_index+1])[test_index]



print('Testing image No.',test_index,'(label=',test_label,')')
print('Our model\'s prediction:')
for i in range(10):
    print('\tlabel =',i,'prob. = {:.4f}%'.format(pred_label[i]))

