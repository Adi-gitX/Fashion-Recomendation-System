# Procedure: How to Run the Recommendation System

This procedure is based on the `fashion_recommendation-System).ipynb` notebook.

The notebook must be run in two stages:
1.  **Indexing:** A one-time process to analyze the dataset and build the search models.
2.  **Inference:** Using the indexed models to get recommendations.

---

## Step 1: Indexing (Training Phase)

You must run this part first. It will process the entire fashion dataset and save the feature files and KNN models.

1.  **Setup:**
    * Open the notebook in a GPU-enabled environment (e.g., Google Colab, Kaggle).
    * Run the initial cells to import libraries (TensorFlow, CLIP, sklearn, etc.).
    * Ensure the **Fashion Recommendation Dataset** is available in your environment (e.g., by adding it from `https://www.kaggle.com/datasets/adityakammati/fashion-recomendation-dataset`).

2.  **Run VGG16 Feature Extraction:**
    * Run the cells under **"3.1 VGG16 Feature Extraction"**.
    * This will process all images and save the visual features as `vgg16_features.npy`.
    * (Note: You can also optionally load the pre-trained VGG16 model from `https://www.kaggle.com/models/adityakammati/vgg16-fashion-recommendation` if you modify the notebook's loading cells).

3.  **Run CLIP Feature Extraction:**
    * Run the cells under **"3.2 CLIP Feature Extraction"**.
    * This will process all images and save the multimodal features as `clip_image_features.npy`.

4.  **Run KNN Index Building:**
    * Run the cells under **"4.1 VGG16 with KNN"** and **"4.2 CLIP with KNN"**.
    * This will load the `.npy` files and train the two separate KNN models.

**Once this is complete, you do not need to run Step 1 again.**

---

## Step 2: Inference (Testing Phase)

After the models are indexed, you can run these cells anytime to get recommendations.

### A. Visual Search (Image-to-Image)

1.  Find the cell containing the `recommend_images_from_image` function.
2.  Change the `test_image_path` variable to point to your test image.
3.  Run the cell.

    ```python
    # Example Usage for Visual Search:
    test_image_path = "/path/to/your/test_image.jpg"
    recommend_images_from_image(test_image_path, n_recommendations=5)
    ```

### B. Semantic Search (Text-to-Image)

1.  Find the cell containing the `recommend_images_from_text` function.
2.  Run the cell. It will provide a text prompt.
3.  Enter your description (e.g., "red floral dress") and press Enter.

    ```python
    # Example Usage for Semantic Search (the cell will prompt you):
    user_query = input("Enter a text description to search images: ")
    recommend_images_from_text(user_query, n_recommendations=5)
    ```
