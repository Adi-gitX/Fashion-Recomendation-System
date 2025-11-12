# Procedure: How to Run the Recommendation System

[cite_start]This procedure is based on the `fashion_recommendation-System).ipynb` notebook [cite: 1-281].

The notebook must be run in two stages:
1.  **Indexing:** A one-time process to analyze the dataset and build the search models.
2.  **Inference:** Using the indexed models to get recommendations.

---

## Step 1: Indexing (Training Phase)

You must run this part first. It will process the entire fashion dataset and save the feature files and KNN models.

1.  **Setup:**
    * Open the notebook in a GPU-enabled environment (e.g., Google Colab, Kaggle).
    * [cite_start]Run the initial cells to import libraries (TensorFlow, CLIP, sklearn, etc.) [cite: 13-27].
    * [cite_start]Ensure the `fashion-product-images-dataset` is available in the environment [cite: 31-33].

2.  **Run VGG16 Feature Extraction:**
    * [cite_start]Run the cells under **"3.1 VGG16 Feature Extraction"** [cite: 40-70].
    * This will process all images and save the visual features as `vgg16_features.npy`.

3.  **Run CLIP Feature Extraction:**
    * [cite_start]Run the cells under **"3.2 CLIP Feature Extraction"** [cite: 71-120].
    * This will process all images and save the multimodal features as `clip_image_features.npy`.

4.  **Run KNN Index Building:**
    * [cite_start]Run the cells under **"4.1 VGG16 with KNN"** and **"4.2 CLIP with KNN"** [cite: 121-140].
    * This will load the `.npy` files and train the two separate KNN models.

**Once this is complete, you do not need to run Step 1 again.**

---

## Step 2: Inference (Testing Phase)

After the models are indexed, you can run these cells anytime to get recommendations.

### A. Visual Search (Image-to-Image)

1.  [cite_start]Find the cell containing the `recommend_images_from_image` function [cite: 141-170].
2.  Change the `test_image_path` variable to point to your test image.
3.  Run the cell.

    ```python
    # Example Usage for Visual Search:
    test_image_path = "/path/to/your/test_image.jpg"
    recommend_images_from_image(test_image_path, n_recommendations=5)
    ```

### B. Semantic Search (Text-to-Image)

1.  [cite_start]Find the cell containing the `recommend_images_from_text` function [cite: 171-281].
2.  Run the cell. It will provide a text prompt.
3.  Enter your description (e.g., "red floral dress") and press Enter.

    ```python
    # Example Usage for Semantic Search (the cell will prompt you):
    user_query = input("Enter a text description to search images: ")
    recommend_images_from_text(user_query, n_recommendations=5)
    ```
