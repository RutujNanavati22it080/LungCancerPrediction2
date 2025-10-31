# 🩺 Lung Cancer Detection Using Convolutional Neural Network (CNN)

### 📘 Project Overview
Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection is essential to improve patient survival rates.  
This project introduces a *deep learning-based approach using Convolutional Neural Networks (CNNs)* to automatically detect lung cancer from *chest X-ray or CT scan images*.

The trained CNN model classifies images as *Normal* or *Cancerous*, providing a reliable, consistent, and automated tool to assist radiologists and clinicians.  
Future work focuses on *segmentation* to highlight cancer regions and *stage detection* to estimate disease progression.

---

## 🧠 Abstract
This project uses CNN architectures like *ResNet, **DenseNet, and **ConvNext* to classify medical images for early lung cancer detection.  
It is trained using the *Kaggle Lung Cancer Dataset*, which contains labeled X-ray and CT scan images.  
The system provides rapid, AI-assisted predictions to support medical professionals.

*Key Features:*
- CNN-based automated classification  
- High accuracy and reduced diagnostic time  
- Future-ready for segmentation and stage estimation  
- Implemented using TensorFlow/Keras and Python  

---

## 🏗 System Architecture

*Architecture Components:*
1. *Input Layer:* Accepts CT/X-ray images  
2. *Preprocessing:* Resizing, normalization, and data augmentation  
3. *CNN Model:* Extracts spatial and hierarchical features  
4. *Classification Layer:* Predicts cancer probability using Sigmoid/Softmax  
5. *Output:* Displays diagnostic result  
6. (Future Scope) *Segmentation Model:* Localizes tumor regions  

---

## ⚙ Methodology

### 🔹 Data Collection
- Dataset: *Kaggle Lung Cancer Dataset*
- Includes labeled X-ray and CT images for cancerous and non-cancerous lungs

### 🔹 Data Preprocessing
- Resizing images to 224×224
- Normalization and pixel scaling  
- Augmentation: rotation, flipping, zooming  

### 🔹 Model Architecture
| Layer | Description |
|-------|--------------|
| Input | 224×224×3 |
| Conv2D | Extracts spatial features |
| MaxPooling2D | Reduces dimensionality |
| Flatten | Converts feature map to vector |
| Dense | Fully connected classifier |
| Activation | ReLU, Sigmoid |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |

### 🔹 Training
- Train-test split: *80% Training / 20% Testing*
- Evaluation Metrics: *Accuracy, Precision, Recall, F1-Score*

---

## 🧩 Technologies Used

| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Image Processing | OpenCV, NumPy |
| Web Framework | Flask / Streamlit |
| Dataset | Kaggle Lung Cancer Dataset |
| Visualization | Matplotlib, Seaborn |

---

## 📊 Results

| Model | Test Accuracy | Test Loss |
|--------|----------------|------------|
| ResNet50 | 0.884 | 0.116 |
| DenseNet165 | 0.845 | 0.155 |
| EfficientNet | 0.794 | 0.206 |
| ConvNext-CNN | *0.913* | *0.087* |

➡ *ConvNext-CNN* achieved the highest performance, demonstrating superior accuracy and stability.

---

## 🚀 Future Enhancements

### 🔸 1. Segmentation
- Integrate *U-Net* or *Mask R-CNN* models  
- Highlight tumor regions with overlay masks on CT scans  

### 🔸 2. Stage Detection
- Estimate tumor size and spread using segmentation data  
- Classify into stages:
  - Stage I – Localized region  
  - Stage II – Moderate region within one lobe  
  - Stage III – Spread across multiple lobes  
  - Stage IV – Extensive metastasis  

---

## 📚 References
1. Baranwal, N. et al. (2021). Classification of Histopathology Images of Lung Cancer Using CNN.  
2. Tang, H. et al. (2019). Automated Pulmonary Nodule Detection using 3D CNN.  
3. Dey, R. et al. (2018). Diagnostic Classification of Lung Nodules Using 3D Neural Networks.  
4. Deepa, P. et al. (2023). An Effective Method for Lung Cancer Classification Using CNN.  
5. Al-Yasriy, H. et al. (2020). Diagnosis of Lung Cancer Based on CT Scans Using CNN.  

---

## 👩‍💻 Contributors
- *Mannat Mangukiya* — [22it071@charusat.edu.in](mailto:22it071@charusat.edu.in)  
- *Rutuj Nanavati* — [22it080@charusat.edu.in](mailto:22it080@charusat.edu.in)  
- *Nishat Shaikh* (Corresponding Author) — [nishatshaikh.it@charusat.ac.in](mailto:nishatshaikh.it@charusat.ac.in)  
- *Jalpesh Vasa* (Guide) — [jalpeshvasa.it@charusat.ac.in](mailto:jalpeshvasa.it@charusat.ac.in)  

---

## 🏁 Conclusion
The proposed CNN-based lung cancer detection system demonstrates strong potential for accurate, fast, and consistent diagnosis from medical imaging data.  
By leveraging deep learning, it reduces manual dependency, minimizes human error, and supports medical professionals in early detection — a key step toward improving survival rates.  
Future integration with segmentation and stage prediction will further enhance diagnostic transparency and clinical value.

---

### 🧬 Keywords
Lung Cancer · CNN · Deep Learning · Medical Imaging · AI · TensorFlow · Image Classification · Flask · Streamlit

---

> ⭐ If you found this project insightful, consider starring the repository!
