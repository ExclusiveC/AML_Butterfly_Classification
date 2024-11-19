# **Butterfly Image Classification Using MobileNetV2**  
![image](https://github.com/user-attachments/assets/575f8583-1203-4a51-a9b8-dd7f601bfa41)

This repository contains the code, data preprocessing scripts, and resources for a deep learning project focused on classifying butterfly species using the MobileNetV2 architecture.  

---

## **Table of Contents**  
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Features](#features)  
4. [Model Architecture](#model-architecture)  
6. [Screenshots](#screenshots)  
7. [Streamlit App Interface](#streamlit-app-interface)  


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
### **Screenshots**:
**Prediction after base_model training:**
![image](https://github.com/user-attachments/assets/16debc24-0de1-439c-8a1a-2b113e62d804)
**Prediction after improved_model training**
![image](https://github.com/user-attachments/assets/adbcddb1-8775-42f3-89f3-888301763c09)


### **Streamlit App Interface**:
**You can try our model in streamlit** https://amlbutterflyclassification-girpaoj9t5wz2ywhfr2mnu.streamlit.app/
![image](https://github.com/user-attachments/assets/962f4c46-1e49-4da6-88f3-21ed31f3b4c2)

