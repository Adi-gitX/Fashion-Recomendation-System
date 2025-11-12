# Multimodal Fashion Recommendation System (VGG16 + CLIP)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-blue?style=for-the-badge&logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?style=for-the-badge&logo=scikitlearn)

This repository contains the code for a dual-model, multimodal fashion recommendation system, as detailed in the accompanying paper, **"Bridging the Semantic and Visual Gap: A Dual-Model System for Multimodal Fashion Recommendation"**.

The system provides two distinct ways to search for fashion items:
1.  **Visual Search (Image-to-Image):** Find items that look visually similar to an image you provide.
2.  **Semantic Search (Text-to-Image):** Find items that match a natural language text description (e.g., "blue striped shirt").

---

## System Architecture

The project implements two parallel pipelines that are indexed and queried separately.

### 1. Pipeline 1: Visual Search (VGG16-CBIR)
This pipeline uses a **VGG16** model (pre-trained on ImageNet) as a feature extractor.
* It generates a **4096-dimensional** feature vector for each image, capturing deep visual attributes like texture, pattern, and shape.
* A **K-Nearest Neighbors (KNN)** model is trained on these vectors to find the closest matches in the database, enabling powerful Content-Based Image Retrieval (CBIR).

### 2. Pipeline 2: Semantic Search (CLIP)
This pipeline uses **OpenAI's CLIP** model, which is trained to understand both text and images in a shared **512-dimensional** embedding space.
* It can translate a text query (like "red floral dress") into a vector.
* It then compares this text vector to the pre-computed image vectors in its own KNN index to find the best semantic matches, bridging the gap between words and images.

---

## Project Resources (Kaggle Hub)

The pre-trained VGG16 model and the dataset are publicly available on Kaggle:

* **Pre-trained VGG16 Model:**
    * **URL:** `https://www.kaggle.com/models/adityakammati/vgg16-fashion-recommendation`

* **Dataset:**
    * **URL:** `https://www.kaggle.com/datasets/adityakammati/fashion-recomendation-dataset`

---

## Project Structure

* **`fashion_recommendation-System).ipynb`**: The complete Jupyter notebook containing all code for:
    * Data loading and preprocessing.
    * **Indexing Pipeline:** Feature extraction for both VGG16 and CLIP.
    * **Inference Pipeline:** Functions for running image-to-image and text-to-image recommendations.
* **`IEEE_Conference_Template__1_.pdf`**: The academic paper detailing the system's architecture, methodology, and results.

---

## How to Use

The notebook is divided into two main phases: **Indexing** (a one-time setup) and **Inference** (for running queries).

For detailed, step-by-step instructions, please see **`HOW_TO_RUN.md`**.
