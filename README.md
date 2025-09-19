# XAI Project

This project implements and benchmarks explainable AI (XAI) methods for image classification, focusing on ensemble and advanced evaluation of Grad-CAM, LIME, and SHAP explanations. The codebase supports preprocessing, model training, prediction, and quantitative benchmarking of explanation quality.

## Project Structure

- `src/`
  - `Benchmarks_ensemble_advanced.ipynb`: Main notebook for benchmarking and ensemble evaluation of XAI methods.
  - `LIME_XAI_implementation.ipynb`: Notebook for generating and saving LIME heatmaps.
  - `SHAP_XAI_implementation.ipynb`: Notebook for generating and saving SHAP heatmaps.
  - `gradcam_XAI_implementation.ipynb`: Notebook for generating and saving GradCAM heatmaps.

  - `model/`
    - `model_training_preprocessing.py`: Data loading and preprocessing pipeline.
    - `model_training.py`, `model_training_tuning.py`: Model training and tuning scripts.
    - `Model_Prediction.py`: Model prediction and evaluation utilities.
  - `custom_lime/LimeExplainer.py`: Custom LIME explainer for image models.
  - `custom_gradcam/GradCAM.py`: Custom Grad-CAM implementation for Keras models (if present).
- `preprocessing/`
  - `preprocessing.py`: Additional preprocessing utilities.
- `outputs/`, `pdf/`: Output folders for results and reports.
- `.conda/`: Conda environment files and dependencies.

## Main Features

- **Data Preprocessing:** Automated loading, augmentation, and normalization of image datasets.
- **Model Training:** Fine-tuning and evaluation of deep learning models (e.g., Xception, InceptionV3).
- **XAI Methods:** Generation of Grad-CAM, LIME, and SHAP heatmaps for model explanations.
- **Ensemble & Benchmarking:** Advanced ensemble methods and quantitative metrics (decision impact ratio, cosine similarity, accordance recall) for comparing XAI methods.
- **Visualization:** Overlay and save heatmaps for interpretability.

## LIME Implementation

- The LIME workflow is implemented in [`src/LIME_XAI_implementation.ipynb`](src/LIME_XAI_implementation.ipynb) and [`src/custom_lime/LimeExplainer.py`](src/custom_lime/LimeExplainer.py).
- The [`LimeExplainer`](src/custom_lime/LimeExplainer.py) class wraps the standard LIME image explainer and adapts it for Keras models and TensorFlow datasets.
- It extracts individual images from the test set, preprocesses them, and generates explanations by perturbing the input image and observing the model's output changes.
- The explainer highlights the most influential superpixels that drive the model's decision, producing both visual overlays and heatmaps.
- Explanations are collected for all test images and stored for further analysis and benchmarking.

## Grad-CAM Implementation

- Grad-CAM is implemented in the benchmarking notebook ([`src/Benchmarks_ensemble_advanced.ipynb`](src/Benchmarks_ensemble_advanced.ipynb)) and/or a custom module (e.g., `src/custom_gradcam/GradCAM.py` if present).
- Grad-CAM computes the gradient of the target class with respect to the output feature maps of a convolutional layer, producing a coarse localization map highlighting important regions in the image.
- The implementation typically involves:
  - Forward-passing an image through the model.
  - Computing gradients of the class score with respect to the feature maps.
  - Weighting the feature maps by the computed gradients and aggregating them.
  - Overlaying the resulting heatmap on the original image for visualization.
- Grad-CAM results are used for qualitative and quantitative benchmarking alongside LIME and SHAP.

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure paths:**
   - Edit `config_server.yaml` or `config.yaml` to set dataset and output directories.

3. **Run notebooks:**
   - Use [`src/LIME_XAI_implementation.ipynb`](src/LIME_XAI_implementation.ipynb) to generate LIME heatmaps.
   - Use [`src/SHAP_XAI_implementation.ipynb`](src/SHAP_XAI_implementation.ipynb) to generate SHAP heatmaps.
   - Use [`src/Benchmarks_ensemble_advanced.ipynb`](src/Benchmarks_ensemble_advanced.ipynb) for benchmarking and ensemble analysis.

## References

- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [LIME](https://arxiv.org/abs/1602.04938)
- [SHAP](https://github.com/slundberg/shap)
---