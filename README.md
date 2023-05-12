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

아래 pseudocode에서는 최상의 결과를 보여주었떤 ResNet50 아키텍처를 사용하여 성능 평가를 한 코드입니다. 
