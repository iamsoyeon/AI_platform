# AI특론
AI 특론 assignment (midterm) + final assignment

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



final assignment (중간과제 코드에서 더 발전시키기)

activation 함수로 Relu 말고, Relu가 갖는 x값이 0보다 작으면 발생하는 Dying ReLU(뉴런이 죽는 현상)현상을 해결하기 위한 leaky_relu를 사용해보면 어떨까 라는 생각을 하게 되었고, 
Conv2D 레이어에서 사용해보았습니다. 
그리고 추가로 3개의 Con2D layer 중, 두번째 레이어의 convolution filter를 64개로 변경해보았고, Maxpooling layer의 pool size를 (2,2)로 통일했습니다. 

(기말과제) 수정한 코드는 Plant_transferlearning_revision_final.ipynb 에서 확인하실 수 있습니다. 

코드의 11번째 셀 [11] 의 Define model architecture 부분이 수정되었습니다.

밑에 pseudocode에서도 수정된 부분을 반영하였으며, <- 표시로 부연설명을 해두었습니다. 

다음은 수정된 코드를 돌린 결과입니다. 


Epoch 1/30
273/273 [==============================] - 284s 1s/step - loss: 0.1024 - accuracy: 0.3455 - val_loss: 0.0607 - val_accuracy: 0.5670
Epoch 2/30
273/273 [==============================] - 280s 1s/step - loss: 0.0582 - accuracy: 0.5912 - val_loss: 0.0420 - val_accuracy: 0.7070
Epoch 3/30
273/273 [==============================] - 281s 1s/step - loss: 0.0464 - accuracy: 0.6784 - val_loss: 0.0336 - val_accuracy: 0.7771
Epoch 4/30
273/273 [==============================] - 273s 1s/step - loss: 0.0394 - accuracy: 0.7383 - val_loss: 0.0270 - val_accuracy: 0.8250
Epoch 5/30
273/273 [==============================] - 271s 996ms/step - loss: 0.0340 - accuracy: 0.7702 - val_loss: 0.0256 - val_accuracy: 0.8293
Epoch 6/30
273/273 [==============================] - 281s 1s/step - loss: 0.0307 - accuracy: 0.7929 - val_loss: 0.0219 - val_accuracy: 0.8618
Epoch 7/30
273/273 [==============================] - 276s 1s/step - loss: 0.0279 - accuracy: 0.8181 - val_loss: 0.0160 - val_accuracy: 0.9042
Epoch 8/30
273/273 [==============================] - 273s 1s/step - loss: 0.0245 - accuracy: 0.8437 - val_loss: 0.0142 - val_accuracy: 0.9158
Epoch 9/30
273/273 [==============================] - 279s 1s/step - loss: 0.0229 - accuracy: 0.8541 - val_loss: 0.0135 - val_accuracy: 0.9239
Epoch 10/30
273/273 [==============================] - 277s 1s/step - loss: 0.0212 - accuracy: 0.8663 - val_loss: 0.0136 - val_accuracy: 0.9234
Epoch 11/30
273/273 [==============================] - 273s 1s/step - loss: 0.0196 - accuracy: 0.8771 - val_loss: 0.0105 - val_accuracy: 0.9453
Epoch 12/30
273/273 [==============================] - 270s 989ms/step - loss: 0.0186 - accuracy: 0.8855 - val_loss: 0.0097 - val_accuracy: 0.9461
Epoch 13/30
273/273 [==============================] - 282s 1s/step - loss: 0.0169 - accuracy: 0.8971 - val_loss: 0.0105 - val_accuracy: 0.9408
Epoch 14/30
273/273 [==============================] - 273s 1000ms/step - loss: 0.0157 - accuracy: 0.9057 - val_loss: 0.0084 - val_accuracy: 0.9562
Epoch 15/30
273/273 [==============================] - 268s 982ms/step - loss: 0.0156 - accuracy: 0.9055 - val_loss: 0.0075 - val_accuracy: 0.9595
Epoch 16/30
273/273 [==============================] - 267s 981ms/step - loss: 0.0149 - accuracy: 0.9126 - val_loss: 0.0061 - val_accuracy: 0.9694
Epoch 17/30
273/273 [==============================] - 267s 979ms/step - loss: 0.0130 - accuracy: 0.9207 - val_loss: 0.0067 - val_accuracy: 0.9629
Epoch 18/30
273/273 [==============================] - 267s 979ms/step - loss: 0.0123 - accuracy: 0.9274 - val_loss: 0.0054 - val_accuracy: 0.9721
Epoch 19/30
273/273 [==============================] - 272s 999ms/step - loss: 0.0127 - accuracy: 0.9253 - val_loss: 0.0063 - val_accuracy: 0.9643
Epoch 20/30
273/273 [==============================] - 270s 991ms/step - loss: 0.0132 - accuracy: 0.9243 - val_loss: 0.0058 - val_accuracy: 0.9702
Epoch 21/30
273/273 [==============================] - 269s 986ms/step - loss: 0.0106 - accuracy: 0.9391 - val_loss: 0.0047 - val_accuracy: 0.9813
Epoch 22/30
273/273 [==============================] - 272s 999ms/step - loss: 0.0117 - accuracy: 0.9329 - val_loss: 0.0071 - val_accuracy: 0.9635
Epoch 23/30
273/273 [==============================] - 270s 989ms/step - loss: 0.0129 - accuracy: 0.9256 - val_loss: 0.0065 - val_accuracy: 0.9658
Epoch 24/30
273/273 [==============================] - 271s 996ms/step - loss: 0.0105 - accuracy: 0.9393 - val_loss: 0.0045 - val_accuracy: 0.9803
Epoch 25/30
273/273 [==============================] - 269s 988ms/step - loss: 0.0094 - accuracy: 0.9472 - val_loss: 0.0033 - val_accuracy: 0.9857
Epoch 26/30
273/273 [==============================] - 273s 1000ms/step - loss: 0.0110 - accuracy: 0.9374 - val_loss: 0.0042 - val_accuracy: 0.9829
Epoch 27/30
273/273 [==============================] - 271s 995ms/step - loss: 0.0098 - accuracy: 0.9464 - val_loss: 0.0040 - val_accuracy: 0.9845
Epoch 28/30
273/273 [==============================] - 267s 979ms/step - loss: 0.0090 - accuracy: 0.9484 - val_loss: 0.0049 - val_accuracy: 0.9729
Epoch 29/30
273/273 [==============================] - 268s 985ms/step - loss: 0.0101 - accuracy: 0.9409 - val_loss: 0.0048 - val_accuracy: 0.9757
Epoch 30/30
273/273 [==============================] - 267s 980ms/step - loss: 0.0093 - accuracy: 0.9485 - val_loss: 0.0029 - val_accuracy: 0.9879


![image](https://github.com/iamsoyeon/AI_platform/assets/83504325/1dddf973-745e-426e-b3cb-64dc682b4115)


val_accuracy나 val_loss를 확인했을 때, 중간과제코드와 드라마틱한 차이는 없었으나, val_accuracy가 약간이라도 더 좋아진 것을 확인할 수 있었습니다. 



2. installation

google colab 에서 실행하였습니다. 

3. run

학습을 위해, plantdisease_transferlearning.ipynb 파일을 실행한다.


4. Pseudocode

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
   - Add a Conv2D layer with 32 filters, a (3, 3) kernel, and 'leaky_relu' activation: model.add(Conv2D(32, (3, 3), activation='leaky_relu', input_shape=input_shape)) <- activation 함수를 relu대신 leaky_relu를 사용
   - Add a MaxPooling2D layer with a (2, 2) pool size: model.add(MaxPooling2D((2, 2))) <- MaxPooling2D layer 의 pool size를 (2,2)로 통일 (기존에는 첫번째 것만 (3,3) size였음.)
   - Add another Conv2D layer with 64 filters and 'leaky_relu' activation: model.add(Conv2D(32, (3,3), activation='leaky_relu'))  <- 이 layer를 64 개의 필터로 수정하였습니다. activation 함수를 relu대신 leaky_relu를 사용
   - Add another MaxPooling2D layer with a (2, 2) pool size: model.add(MaxPooling2D((2, 2)))
   - Add a Conv2D layer with 64 filters and 'leaky_relu' activation: model.add(Conv2D(64, (3, 3), activation='leaky_relu')) <- activation 함수를 relu대신 leaky_relu를 사용
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



