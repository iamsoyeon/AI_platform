# AI_platform
AI platform assignment (midterm) 

1. introduction 

https://github.com/abhinavsagar/plant-disease
APA: Sagar, A., & Dheeba, J. (2020). On Using Transfer Learning For Plant Disease Detection. bioRxiv.

제가 선정한 논문은 전위학습을 통한 plant - disease detection 을 수행한 연구입니다. 
이 논문에서, 신경망이 이미지 분류의 맥락에서 식물 질병 인식에 어떻게 사용될 수 있는지를 보여줍니다.
데이터셋으로는 38가지 종류의 질병을 가진 공개적으로 이용 가능한 plant village 데이터 세트를 사용했다. 
작업의 백본으로 VGG16, ResNet50, InceptionV3, InceptionResNet 및 DenseNet169를 포함한 다섯 가지 아키텍처를 비교했고, ResNet50이 최상의 결과를 보여주었다.
평가를 위해 accuracy, precision, recall, F1 score 및 등급별 혼동 메트릭을 사용했습니다. 

아래 pseudocode는 최상의 결과를 보여주는 ResNet50 아키텍처를 사용하여 성능 평가를 한 코드입니다. 
해당 모델의 code는 위 깃허브링크의 code -> part1.ipynb 에서 확인할 수 있습니다. 

해당 코드의 일부를 수정하였으며, 수정하여 실행한 코드는 PlantdiseaseMtransferlearning.ipynb 파일에 있습니다.


2. installation

google colab 에서 실행하였습니다. 

3. run

학습을 위해, plantdisease_transferlearning.ipynb 파일을 실행한다.


Pseudocode

Get a clone https://github.com/spMohanty/PlantVillage-Dataset.git

Change the current directory to PlantVillage-Dataset/data_distribution_for_SVM

Import warnings, os, glob, matplotlib.pyplot, keras

Set the directory paths:
train_dir ="./train/"  Path to the training directory
test_dir="./test/"    Path to the testing directory

Define get_files(directory):   takes a directory path as input and returns the count of files in that directory
	Initialize a count variable to 0
	for current_path,dirs,files in os.walk(directory) : to iterate over the directory and its subdirectories
		for dr in dirs:
			increment the count by the number of files in each subdirectory
			for each subdirectory, use `glob.glob()` to get the list of files.
	return count

Get the number of classes and the number of train and test images:
   train_samples = get_files(train_dir)       
 	num_classes = len(glob.glob(train_dir + "/*"))
   test_samples = get_files(test_dir)
print the number of classes, train images, and test images.

Create an ImageDataGenerator for normalization for the test data:
test_datagen = ImageDataGenerator(rescale=1./255)

Define the image dimensions, input shape, and batch size:
   - img_width, img_height = 256, 256
   - input_shape = (img_width, img_height, 3)
   - batch_size = 32
Create an image data generator for learning:
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=batch_size)

Create the model: 
	import Sequential, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
	Initialize a Sequential model: model = Sequential()
   - Add a Conv2D layer with 32 filters, a (3, 3) kernel, and 'relu' activation: model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
   - Add a MaxPooling2D layer with a (3, 3) pool size: model.add(MaxPooling2D((3, 3)))
   - Add another Conv2D layer with 32 filters and 'relu' activation: model.add(Conv2D(32, (3,3), activation='relu'))
   - Add another MaxPooling2D layer with a (2, 2) pool size: model.add(MaxPooling2D((2, 2)))
   - Add a Conv2D layer with 64 filters and 'relu' activation: model.add(Conv2D(64, (3, 3), activation='relu'))
   - Add another MaxPooling2D layer with a (2, 2) pool size: model.add(MaxPooling2D((2, 2)))
   - Flatten the output of the previous layer: model.add(Flatten())
  	- Add a Dense layer with 512 units and 'relu' activation: model.add(Dense(512,activation='relu'))
   - Add a Dropout layer with a dropout rate of 0.5: model.add(Dropout(0.5))
   - Add a Dense layer with 128 units and 'relu' activation: model.add(Dense(128,activation='relu'))
   - Add a Dense layer with the number of classes as the number of units and 'softmax' activation: model.add(Dense(num_classes, activation='softmax'))
	Summary the model: model.summary()

Define the model_layers using the layer names of the model:
model_layers = [layer.name for layer in model.layers]

Create separate models to get the output of specific layers in the model:
conv2d_3_output: Model with inputs=model.input and outputs=model.get_layer('conv2d').output
max_pooling2d_3_output: Model with inputs=model.input and outputs=model.get_layer('max_pooling2d').output
conv2d_4_output: Model with inputs=model.input and outputs=model.get_layer('conv2d_1').output
max_pooling2d_4_output: Model with inputs=model.input and outputs=model.get_layer('max_pooling2d_1').output
conv2d_5_output: Model with inputs=model.input and outputs=model.get_layer('conv2d_2').output
max_pooling2d_5_output: Model with inputs=model.input and outputs=model.get_layer('max_pooling2d_2').output
flatten_1_output: Model with inputs=model.input and outputs=model.get_layer('flatten').output

Obtain features from the intermediate models: 
conv2d_3_features : conv2d_3_output.predict(img)
max_pooling2d_3_features : max_pooling2d_3_output.predict(img)
conv2d_4_features : conv2d_4_output.predict(img)
max_pooling2d_4_features : max_pooling2d_4_output.predict(img)
conv2d_5_features : conv2d_5_output.predict(img)
max_pooling2d_5_features : max_pooling2d_5_output.predict(img)
flatten_1_features : flatten_1_output.predict(img)

Visualize the features using subplots and imshow: 
	fig = plt.figure with figsize=(14, 7)
	set columns = 8, rows = 4.
	for i in range(columns * rows) : 
		fig.add_subplot(rows, columns, i+1)
   		plt.axis('off')
    		plt.title('filter' + str(i))
  		plt.imshow(conv2d_3_features[0, :, :, i], cmap='viridis')

Create a validation generator using flow_from_directory

Compile the model:
	import Adam
	Use the Adam optimizer with a learning rate of 0.001: optimizer = Adam(lr=0.001)
	Compile the model with 'binary_crossentropy' loss and 'accuracy' metric

Train the model:
	Use the `fit_generator` method of the model to train on the train generator data:
 		‘train = model.fit_generator’ with epoch = 30, steps_per_epoch=train_generator.samples, validation_data=validation_generator

Extract training history for accuracy and loss: 
	extract acc 
	extract val_acc 
	extract loss
	extract val_loss
Set epochs range from 1 to len(acc) + 1
Plot the training and validation accuracy
Plot the training and validation loss



