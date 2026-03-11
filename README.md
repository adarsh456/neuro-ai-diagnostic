# 🧠 Neuro-AI Diagnostic Dashboard
**Advanced MRI Classification & XAI Tumor Localization**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🌐 Live Demo
**Try the web app here:** (https://neuro-ai-diagnostic-fcluivdhgflipfkfo5fggh.streamlit.app/)

---

## 📖 About the Project
The Neuro-AI Diagnostic Dashboard is a deep learning-powered medical imaging tool designed to assist in the classification and localization of brain tumors from MRI scans. 

Going beyond standard classification, this project implements **Explainable AI (XAI)** using **Grad-CAM** (Gradient-weighted Class Activation Mapping). This allows the model to not only predict the type of tumor but also generate a visual heatmap superimposed over the original MRI, "explaining" exactly which regions of the brain the AI focused on to make its diagnosis.

### 🔬 Key AI Capabilities
* **Glioma Detection**
* **Meningioma Detection**
* **Pituitary Tumor Detection**
* **Healthy Brain Verification**

---

## 📸 App Interface
<img width="1920" height="914" alt="Screenshot 2026-03-12 031658" src="https://github.com/user-attachments/assets/e87b19be-b196-4d99-b647-aa7f93fcd466" />





---

## 🛠️ Technology Stack
* **Frontend/Deployment:** Streamlit Community Cloud
* **Machine Learning:** TensorFlow & Keras (Convolutional Neural Networks)
* **Explainable AI:** Grad-CAM & OpenCV
* **Data Processing:** NumPy, Pillow (PIL)
* **Hosting Workaround:** GitHub Releases (used for hosting the large `.h5` model file to bypass standard file size limits).

---

