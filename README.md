# Multimodal Fashion Recommendation System (VGG16 + CLIP)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-blue?style=for-the-badge&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)


## 🎥 Demo Video  
[![Watch the demo](https://img.shields.io/badge/Watch%20Demo-Click%20Here-red?style=for-the-badge&logo=youtube)](https://youtu.be/0QEFfEm3eLA)

This repository contains the code for a dual-model, multimodal fashion recommendation system, as detailed in the accompanying paper, **"Bridging the Semantic and Visual Gap: A Dual-Model System for Multimodal Fashion Recommendation"**.

The system provides two distinct ways to search for fashion items:
1. **Visual Search (Image-to-Image):** Find items that look visually similar to an image you provide.
2. **Semantic Search (Text-to-Image):** Find items that match a natural language text description (for example, "blue striped shirt").

---

## System Architecture

The project implements two parallel pipelines that are indexed and queried separately.

### 1. Pipeline 1: Visual Search (VGG16-CBIR)  
This pipeline uses a **VGG16** model (pre-trained on ImageNet) as a feature extractor.  
* It produces a **4096-dimensional** feature vector for each image, capturing pattern, texture and shape.  
* A **K-Nearest Neighbors (KNN)** model is trained on these vectors to find the closest matches in the database, enabling Content-Based Image Retrieval (CBIR).

### 2. Pipeline 2: Semantic Search (CLIP)  
This pipeline uses **OpenAI’s CLIP** model, which understands both images and text in a shared **512-dimensional** embedding space.  
* A text query (for example, "red floral dress") is converted into a vector.  
* That vector is compared with the pre-computed CLIP image vectors via a KNN index to find the top semantic matches.

---

## Project Resources (Kaggle Hub)

* **Pre-trained VGG16 Model:**  
  `https://www.kaggle.com/models/adityakammati/vgg16-fashion-recommendation`

* **Dataset:**  
  `https://www.kaggle.com/datasets/adityakammati/fashion-recomendation-dataset`

---

## Project Structure

* **`fashion_recommendation-System-Traning.ipynb`**  
  Training notebook for both pipelines (VGG16 + CLIP).

* **`fashion_recommendation_testing_notebook.ipynb`**  
  Notebook for running both visual and semantic search inference.

* **`fashion-recomendation-Research-Paper.pdf`**  
  The full research paper explaining architecture, methodology and results.

* **`HOW_TO_RUN.md`**  
  Step-by-step instructions for using both pipelines.

---

## How to Use

The workflow is split into two phases:

### **Indexing Phase (one-time setup)**  
* Extract features using VGG16 and CLIP.  
* Build KNN indices for both pipelines.

### **Inference Phase**  
* Run **image-to-image** search using VGG16.  
* Run **text-to-image** search using CLIP.

For full setup instructions, see **`HOW_TO_RUN.md`.**
