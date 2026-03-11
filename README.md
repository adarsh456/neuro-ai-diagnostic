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
*(Upload your dashboard screenshot to your GitHub repo, click on it, copy the image link, and replace the URL below!)*

![Dashboard Screenshot](INSERT_LINK_TO_YOUR_SCREENSHOT_HERE)

---

## 🛠️ Technology Stack
* **Frontend/Deployment:** Streamlit Community Cloud
* **Machine Learning:** TensorFlow & Keras (Convolutional Neural Networks)
* **Explainable AI:** Grad-CAM & OpenCV
* **Data Processing:** NumPy, Pillow (PIL)
* **Hosting Workaround:** GitHub Releases (used for hosting the large `.h5` model file to bypass standard file size limits).

---

## 💻 How to Run Locally<img width="1893" height="941" alt="Screenshot 2026-03-12 025257" src="https://github.com/user-attachments/assets/c3eadac7-56ef-48db-a397-0f225964ddc4" />


1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
