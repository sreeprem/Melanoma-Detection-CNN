# Melanoma-Detectio
Melanoma Detection
# Melanoma Skin Cancer Detection
This project aims to develop an automated classification system for detecting melanoma skin cancer using image processing techniques. Melanoma is the deadliest form of skin cancer, and early detection is crucial for successful treatment. The project utilizes Convolutional Neural Networks (CNN) to classify skin lesions and improve the diagnostic process.


## Abstract
In cancer, there are over 200 different forms, and melanoma is considered the deadliest type of skin cancer. The current diagnostic procedure involves clinical screening, dermoscopic analysis, and histopathological examination. However, this process is time-consuming and can take a week or more. The project's objective is to develop a predictive model to shorten the diagnosis time to just a couple of days by leveraging advanced image classification techniques.


## Problem Statement
The current gap in the diagnostic process, from dermatologist appointment to biopsy report, takes a considerable amount of time. The project aims to address this issue by providing a predictive model that uses a CNN to classify nine types of skin cancer based on lesion images. The reduction in diagnosis time has the potential to positively impact millions of people.


## Motivation
The primary motivation behind this project is to contribute to the efforts of reducing deaths caused by skin cancer. By leveraging computer vision and machine learning techniques, the project aims to utilize advanced image classification technology for the betterment of people's well-being. The advancements in machine learning and deep learning, specifically in the field of computer vision, provide scalable solutions across various domains.

## Dataset
The dataset used for training and evaluation consists of 2357 images of malignant and benign oncological diseases. These images were obtained from the International Skin Imaging Collaboration (ISIC). The dataset was sorted according to the classifications provided by ISIC, and subsets were created with an equal number of images for each class.

## CNN Architecture Design
To achieve higher accuracy in skin cancer classification, a custom CNN model was designed. The architecture includes the following layers:

Rescaling Layer - To rescale the input image from the [0, 255] range to the [0, 1] range.
Convolutional Layer - Applies convolution operations to the input, reducing image size and extracting features.
Pooling Layer - Reduces feature map dimensions, summarizing the features.
Dropout Layer - Helps prevent overfitting by randomly setting input units to 0 during training.
Flatten Layer - Converts the output of convolutional layers into a 1-dimensional array.
Dense Layer - A fully-connected neural network layer that receives input from the previous layer.
Activation Functions - ReLU (Rectified Linear Unit) for hidden layers and Softmax for the output layer.

## Model Evaluation
The model's performance was evaluated using various metrics, such as accuracy, precision, recall, and F1 score. The evaluation results provide insights into the model's effectiveness in classifying skin cancer based on lesion images.


## Conclusion
This project demonstrates the potential of image processing and CNN models in automating the classification of melanoma skin cancer. By reducing the diagnostic time, it has the potential to positively impact the lives of individuals affected by skin cancer. However, further research and improvements can be made to enhance the accuracy and efficiency of the model.

