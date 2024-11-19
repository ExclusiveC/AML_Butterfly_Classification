# **Butterfly Image Classification Using MobileNetV2**  
![image](https://github.com/user-attachments/assets/575f8583-1203-4a51-a9b8-dd7f601bfa41)

This repository contains the code, data preprocessing scripts, and resources for a deep learning project focused on classifying butterfly species using the MobileNetV2 architecture.  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Features](#features)  
4. [Model Architecture](#model-architecture)  
6. [Usage](#usage)  
7. [Results](#results)  


---

## **Project Overview**  
This project aims to classify 75 species of butterflies using a convolutional neural network (CNN) based on the MobileNetV2 architecture. It demonstrates preprocessing, training, and evaluation techniques for achieving a validation accuracy of **87.08%**.  

**Key Objectives:**  
- Automate butterfly species identification.  
- Address challenges like class imbalance and varying image quality.  
- Provide insights into deep learning techniques for image classification.  

---

## **Dataset**  
**Source:** [Kaggle Butterfly Dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)  
![image](https://github.com/user-attachments/assets/273ee348-4d37-477e-a2fe-337c0c7c57a8)

### Dataset Description:  
- **Total Images:** 6,499 images of 75 butterfly species.  
- **Train/Test Split:**  
  - **Training:** 80%  
  - **Validation:** 20%  
- **Preprocessing Steps:**  
  - Resizing images to 224x224.  
  - Normalizing pixel values.  
  - Data augmentation (rotations, flips, zooms).  

---

## **Features**  
1. **MobileNetV2-based Classifier**  
2. **Data Augmentation** for improved generalization.  
3. **Dropout Layers** to combat overfitting.  
4. **Accuracy and Loss Visualization** during training.  

---

## **Model Architecture**  
The model utilizes MobileNetV2 pretrained on ImageNet, with added dense layers and Dropout for butterfly-specific classification.  

### Architecture Overview:  
- **Base Model:** MobileNetV2 (pretrained weights).  
- **Modified Layers:**  
  - Flatten and Dense layers for classification.  
  - Dropout layers for regularization.  
- **Output:** Softmax activation with 75 classes.  


![image](https://github.com/user-attachments/assets/adbcddb1-8775-42f3-89f3-888301763c09)


