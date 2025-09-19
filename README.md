# XAI Project

This project implements and benchmarks explainable AI (XAI) methods for image classification, focusing on ensemble and advanced evaluation of Grad-CAM, LIME, and SHAP explanations. The codebase supports preprocessing, model training, prediction, and quantitative benchmarking of explanation quality.

## Project Structure

- `src/`
  - `Benchmarks_ensemble_advanced.ipynb`: Main notebook for benchmarking and ensemble evaluation of XAI methods.
  - `SHAP_XAI_implementation.ipynb`: Notebook for generating and saving SHAP heatmaps.
  - `model/`
    - `model_training_preprocessing.py`: Data loading and preprocessing pipeline.
    - `model_training.py`, `model_training_tuning.py`: Model training and tuning scripts.
    - `Model_Prediction.py`: Model prediction and evaluation utilities.
  - `custom_lime/LimeExplainer.py`: Custom LIME explainer for image models.
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

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure paths:**
   - Edit `config_server.yaml` to set dataset and output directories.

3. **Run notebooks:**
   - Use `src/SHAP_XAI_implementation.ipynb` to generate SHAP heatmaps.
   - Use `src/Benchmarks_ensemble_advanced.ipynb` for benchmarking and ensemble analysis.

## References

- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [LIME](https://arxiv.org/abs/1602.04938)
- [SHAP](https://github.com/slundberg/shap)

---

For questions or contributions, please open an issue or pull request.