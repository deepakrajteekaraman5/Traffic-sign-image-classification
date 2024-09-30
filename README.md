# **Traffic Sign Classification using German Traffic Sign Recognition Benchmark (GTSRB)**

## **Overview**

This repository contains a traffic sign classification project built using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The goal of this project is to classify traffic signs from images into their respective categories using deep learning techniques.

## **Dataset**

The dataset used for this project is obtained from the [GTSRB Kaggle dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). It consists of more than 50,000 images of 43 different classes of traffic signs.

### **Key features of the dataset:**
- Images are of various sizes and resolutions.
- Dataset divided into 43 distinct categories of traffic signs.
- High variability in lighting, angles, and occlusions in the images.

## **Project Files**

- `Traffic_sign_image_classification.ipynb`: Jupyter Notebook containing the complete code for training, validating, and testing the traffic sign classification model.
- `README.md`: This file provides an overview of the project, setup instructions, and usage guidelines.

## **Installation**

To run this project locally, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone <repository-link>

2. **Navigate to the project directory**:
    ```bash
   cd traffic-sign-classification

3. **Install the required dependencies**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt

4. **Download the dataset**:
   The dataset used is available on Kaggle. Download the dataset and place it in the appropriate directory.
5. **Dataset Placement**:
After downloading the dataset from Kaggle, place the dataset folder in the following directory within your project:
   ```bash
   /content/drive/MyDrive/your_project_folder/
##  **Usage**
Once you have installed all the dependencies, follow the steps below to run the project:
Open the Jupyter Notebook:
   ```bash
 jupyter notebook
  ```
**Run the notebook**:
Navigate to Traffic_sign_image_classification.ipynb and execute the cells to train the traffic sign classification model. Ensure that the dataset is placed in the correct directory as expected by the notebook.

## **Model Architecture**
The model architecture used for traffic sign classification is a Convolutional Neural Network (CNN) with multiple layers of convolutions, batch normalization, and pooling. The model is trained to classify traffic signs into 43 categories. Below is a detailed explanation of the model and its training process.

1. **Convolutional Layers**:
The model starts with a convolution block that extracts features from the input images.
Each convolution block is followed by Batch Normalization to normalize the activations and improve the training process.
Max Pooling layers are used to reduce the spatial dimensions of the feature maps, making the network computationally efficient.
Dropout layers are used to prevent overfitting by randomly setting a fraction of the input units to 0 during training.

2. **Conv Block 1**:
Two Conv2D layers with 32 filters and a kernel size of (5x5) are used to detect local features in the image.
BatchNormalization is applied after each convolution to normalize the feature maps.
MaxPooling with a pool size of (2x2) is used to reduce the spatial dimensions of the feature maps.
Dropout with a rate of 0.25 is used to reduce overfitting by randomly dropping units during training.

3. **Conv Block 2**:
Two more Conv2D layers with 64 filters and a kernel size of (3x3) are used to extract deeper features.
BatchNormalization and MaxPooling are again applied in this block.
Dropout with a rate of 0.25 is used again to regularize the model.

4. **Global Average Pooling**:
Instead of flattening the feature maps after the convolution blocks, GlobalAveragePooling2D is used to downsample the feature maps, which results in a more compact representation and helps avoid overfitting.

5. **Fully Connected Layer**:
A Dense layer with 256 units is used to connect the features from the convolutional layers to the output.
Dropout with a rate of 0.3 is applied to this fully connected layer.

6. **Output Layer**:
The output layer consists of 43 units (corresponding to the number of traffic sign classes) with a softmax activation function, which converts the model output to a probability distribution over the classes.

7. **Compilation**:
The model is compiled using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.9 to accelerate convergence and reduce oscillations. The loss function used is categorical cross-entropy, as this is a multi-class classification problem.
   ```python
   optimizer = SGD(learning_rate=0.01, momentum=0.9)
   model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
   ```
9. **Data Augmentation**:
To improve generalization and reduce overfitting, ImageDataGenerator is used for real-time data augmentation. This technique helps the model learn robust features by providing slightly altered versions of the same images during training. The transformations applied are:
   **Rotation**: Random rotations between -15° and +15°.
   **Zoom**: Random zooming between 80% and 120%.
   **Width/Height Shift**: Randomly shift images horizontally and vertically by 10% of the image dimensions.
   **Shear**: Random shearing transformations.
   **Horizontal Flip**: Random horizontal flips to create mirrored versions of the images.
    ```python
    datagen = ImageDataGenerator(
    rotation_range=15,       # Random rotations between -15° and +15°
    zoom_range=0.2,          # Random zooming between 80% and 120%
    width_shift_range=0.1,   # Random horizontal shifts (10% of total width)
    height_shift_range=0.1,  # Random vertical shifts (10% of total height)
    shear_range=0.1,         # Random shearing
    horizontal_flip=True,    # Randomly flip images horizontally
    vertical_flip=False      # Vertical flipping if needed
    )
    ```
    This process augments the training dataset, providing more diverse training samples without actually increasing the size of the dataset.

## **Model Performance**
During training, the model's performance was monitored using two metrics: accuracy and loss. The model was trained for 20 epochs, and the results are summarized in the plots below.

1. **Accuracy Plot**
The accuracy plot shows the model's training and validation accuracy over 20 epochs.

![Accuracy](https://github.com/user-attachments/assets/e7ef8bfe-5967-417a-8322-8661afa639a6)

As seen in the plot, both training and validation accuracy increase rapidly during the first few epochs and stabilize around epoch 5. The model achieves high accuracy on both the training and validation sets, indicating that the model is effectively learning from the data and generalizing well to unseen validation data.
By the end of training, the accuracy is very close to 100%, showing that the model is capable of correctly classifying almost all traffic signs in the dataset.

2. **Loss Plot**
The loss plot shows the training and validation loss over 20 epochs.

![Loss](https://github.com/user-attachments/assets/7cd3da2e-703a-4595-97ce-28dedf9483a8)

In this plot, we see a sharp decrease in loss during the initial epochs, followed by stabilization. Both training and validation loss decrease as the model learns, indicating that the model is improving its predictions over time. The gap between the training and validation loss is minimal, suggesting that the model is not overfitting and generalizes well to the validation data.
The final loss values indicate that the model’s predictions are both accurate and confident, given the low values of both training and validation loss by the end of the training process.

## **Challenges**
1. **Lighting and Occlusions**:
Traffic signs appear in different lighting conditions and may be partially blocked by objects. The model may still struggle despite using data augmentation techniques like rotation and zoom.

2. **Similar Signs**:
Some traffic signs have similar shapes but different meanings (e.g., speed limits with different numbers). The model might confuse these subtle differences.

3. **Small Datasets in Some Classes**:
Some classes have fewer examples, leading to overfitting. Techniques like Dropout, L2 regularization, and data augmentation help, but class imbalance remains a challenge.




