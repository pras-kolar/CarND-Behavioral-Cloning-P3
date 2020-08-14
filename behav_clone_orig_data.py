import csv
import cv2

import numpy as np
from scipy import ndimage

from sklearn.model_selection import train_test_split

def process_image(img):
	print("In process image")


"""    
# Code to combine Udacity and PK data
lines = []
my_data = ['/home/workspace', '/home/workspace/datafiles/erratic_correct', '/home/workspace/datafiles/erratic_correct2', '/home/workspace/datafiles/erratic_correct3']
udacity_data = ['/home/workspace/datafiles/datas']

for list in [my_data, udacity_data]:
    print("List Values", list)
    for filename in list:
        with open(filename + '/driving_log.csv') as file:
            for line in csv.reader(file):
                if 'IMG' in line[0]:
                    lines.append(line)
"""
lines = []
#with open('../0806/driving_log.csv') as csvfile:
#with open('../good/driving_log.csv') as csvfile:
#with open('../sim_full_good/driving_log.csv') as csvfile:
with open('../orig_data/data/driving_log.csv') as csvfile:
#with open('../training_data/sim_ml/driving_log.csv') as csvfile:    
	reader = csv.reader(csvfile)
	next(reader)
	print("B4 for")
	for line in reader:
		lines.append(line)
		#print("Lines size : ", len(lines))     

images = []
car_images = []
images_center = []
images_left = []
images_right = []
measurements = []

steering_angles = []
steerings_centered = []
steerings_left = []
steerings_right = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	#print("File : ", filename)
	#current_path = '../0806/IMG/' + filename
	#image_path = '../0806/IMG/'
	#current_path = '../good/IMG/' + filename
	#image_path = '../good/IMG/'
	#current_path = '../sim_full_good/IMG/' + filename
	#image_path = '../sim_full_good/IMG/'
	current_path = '../orig_data/data/IMG/' + filename
	image_path = '../orig_data/data/IMG/'
	#current_path = '../training_data/sim_ml/IMG/' + filename
	#print("Current Path : ", current_path)
	image = cv2.imread(current_path)
	#image = ndimage.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	#print("File : Measurement : ", filename, measurement)

	steering_center = float(line[3])

	# create adjusted steering measurements for the side camera images
	#correction = 0.4 # this is a parameter to tune
	correction = 0.6 # this is a parameter to tune
	steering_left = steering_center + correction + 0.2
	steering_right = steering_center - correction
#	print(steering_left, steering_right)
	steerings_centered.append(steering_center)
	steerings_left.append(steering_left)
	steerings_right.append(steering_right)
	
	# read in images from center, left and right cameras
	#path = '../IMG/' # fill in the path to your training IMG directory
	#center_cam_fname = line[0].split('/')[-1]
	#print("Center - ", center_cam_fname)
	#print("File : ", image_path + line[0].split('/')[-1])
	#print("Center File : ", line[0].split('/')[-1])
	img_center = cv2.imread(image_path + line[0].split('/')[-1])
	img_left = cv2.imread(image_path + line[1].split('/')[-1])
	img_right = cv2.imread(image_path + line[2].split('/')[-1])
	
	#print("img_center : ", img_center.shape)
	#print("img_left   : ", img_left.shape)
	#print("img_right  : ", img_right.shape)
    
	images_center.append(img_center)
	images_left.append(img_left)
	images_right.append(img_right)

	car_images.extend([img_center, img_left, img_right])
	steering_angles.extend([steering_center, steering_left, steering_right])
    
	"""
	img_center = process_image(np.asarray(image.open(path + line[0])))
	img_left = process_image(np.asarray(image.open(path + line[1])))
	img_right = process_image(np.asarray(image.open(path + line[2])))
	"""
augmented_images = []
augmented_measurements = []

for image, measurement in zip(car_images, steering_angles):
#for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

#print("augmented_images : ", len(augmented_images))
X = np.array(augmented_images)
y = np.array(augmented_measurements)
#print("X.shape", len(augmented_images))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train = X_train.reshape(X_train.shape[0], 160,320,3)

from sklearn.preprocessing import StandardScaler

#print("X_train.shape", X_train.shape)

#nsamples, nx, ny, num = X_train.shape # Comment for now pk
#X_train = X_train.reshape((nsamples,nx*ny*num))

###### Review this again ############
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from matplotlib import pyplot


# Initialising the ANN
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# Crop the images before processing in the NN
model.add(Cropping2D(cropping=((50,17), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))

model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Dense(120, activation = 'relu'))

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(84, activation = 'relu'))

model.add(Dense(62, activation = 'relu')) #Added now

# Adding the input layer and the first hidden layer
#model.add(Dense(32, activation = 'relu', input_dim = 8))
model.add(Dense(32, activation = 'relu'))
"""
# Adding the second hidden layer
model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(0.75, input_shape=(4,)))

# Adding the third hidden layer
model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(0.75, input_shape=(4,)))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

model.add(Dropout(0.5, input_shape=(4,)))

"""
# Adding the output layer
model.add(Dense(1))
#model.add(Dropout(0.5, input_shape=(4,)))
model.add(Dropout(0.5, input_shape=(4,)))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7, )

model.save('sim_3cam_0808_ucity_data.h5')

# Reference "https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/"
# Reference "https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/"
