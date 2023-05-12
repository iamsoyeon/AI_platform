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


2. installation

google colab 에서 실행하였습니다. 

3. run 



Pseudocode

1. Clone the PlantVillage-Dataset repository from GitHub.
2. Change the current directory to PlantVillage-Dataset/data_distribution_for_SVM.
3. Import the necessary libraries and modules.

4. Define the following variables:
   - train_dir: Path to the training directory
   - test_dir: Path to the testing directory
   - img_width, img_height: Image width and height
   - batch_size: Batch size for training

5. Define a function named "get_files" that takes a directory path as input and returns the count of files in that directory.

6. Calculate the following:
   - train_samples: Number of training images using the "get_files" function on the train_dir
   - num_classes: Number of classes by counting the subdirectories in train_dir
   - test_samples: Number of testing images using the "get_files" function on the test_dir

7. Print the number of classes, train images, and test images.

8. Create an ImageDataGenerator for training data:
   - Rescale the pixel values to a range of [0, 1]
   - Apply shear range and zoom range for data augmentation
   - Set the validation split to 0.2 for creating a validation set
   - Enable horizontal flipping

9. Create an ImageDataGenerator for testing data:
   - Rescale the pixel values to a range of [0, 1]

10. Create a train_generator using the train_datagen.flow_from_directory() method:
    - Provide the train_dir and target size
    - Set the batch size

11. Create a test_generator using the test_datagen.flow_from_directory() method:
    - Provide the test_dir and target size
    - Set the batch size

12. Define the model architecture using the Sequential API:
    - Add a Conv2D layer with 32 filters, a (3, 3) kernel, and 'relu' activation function, input shape is (img_width, img_height, 3)
    - Add a MaxPooling2D layer with a (3, 3) pool size
    - Add another Conv2D layer with 32 filters and a (3, 3) kernel
    - Add another MaxPooling2D layer with a (2, 2) pool size
    - Add a Conv2D layer with 64 filters and a (3, 3) kernel
    - Add another MaxPooling2D layer with a (2, 2) pool size
    - Add a Flatten layer
    - Add a Dense layer with 512 units and 'relu' activation function
    - Add a Dropout layer with a dropout rate of 0.5
    - Add a Dense layer with 128 units and 'relu' activation function
    - Add a Dense layer with 38 units (number of classes) and 'softmax' activation function

13. Print the model summary.

14. Create separate models to get the output of specific layers in the model:
    - conv2d_3_output: Model with inputs=model.input and outputs=model.get_layer('conv2d').output
    - max_pooling2d_3_output: Model with inputs=model.input and outputs=model.get_layer('max_pooling2d').output
    - conv2d_4_output: Model with inputs=model.input and outputs=model.get_layer('conv2d_1').output
    - max_pooling2d_4_output: Model with inputs=model.input and outputs=model.get_layer('max_pooling2d_1').output
    - conv2d_5_output: Model with inputs=model.input and outputs=model




