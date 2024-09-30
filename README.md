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

**Run the notebook**:
Navigate to Traffic_sign_image_classification.ipynb and execute the cells to train the traffic sign classification model. Ensure that the dataset is placed in the correct directory as expected by the notebook.



