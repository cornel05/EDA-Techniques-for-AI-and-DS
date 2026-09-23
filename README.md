# EDA Techniques for AI and Data Science

Exploratory data analysis and a classical machine learning pipeline across
four data modalities, built for the Programming for Artificial Intelligence
and Data Science course (P4AI-DS, CO3135) at Ho Chi Minh City University of
Technology.

## Datasets

- **Tabular** — Advertisement Click on Ad, 1,000 rows.
- **Text** — Stanford Question Answering Dataset (SQuAD 1.1).
- **Image** — Street View House Numbers (SVHN).
- **Multimodal** — Flickr8k (images with captions).

## Live Dashboard

[cornel05.github.io/EDA-Techniques-for-AI-and-DS](https://cornel05.github.io/EDA-Techniques-for-AI-and-DS/)

The dashboard links to per-modality EDA and ML report pages:

- `index.html` — landing page
- `tabular.html` / `tabular_ml.html` — Tabular EDA / ML analysis
- `text.html` / `text_ml.html` — Text EDA / ML analysis
- `image.html` / `image_ml.html` — Image EDA / ML analysis
- `multimodal.html` — Multimodal EDA

## Pipeline

The tabular track (`eda_tabular.ipynb` → `modeling_tabular.ipynb` →
`evaluation_tabular.ipynb`) follows an end-to-end EDA-driven workflow,
documented in `notebooks/STUDY_GUIDE.md`:

1. EDA-informed preprocessing (missing values, outliers, target-signal
   ranking) feeding a scikit-learn `Pipeline` / `ColumnTransformer`.
2. Cross-validated comparison across a model zoo (linear, tree-based, and
   neural baselines).
3. Hyperparameter tuning on the selected model.
4. Held-out evaluation: ROC/PR curves, calibration, threshold tuning,
   learning curves.
5. Permutation importance and SHAP explanations.
6. Error analysis by slice.

The image track (`image_ML.ipynb`) runs a parallel comparison: classical
classifiers (Logistic Regression, Linear SVM, Random Forest, MLP, Gaussian
Naive Bayes) on pretrained CNN features (ResNet-18, MobileNetV2) and raw
pixels, plus fine-tuning of ResNet-18 and MobileNetV2. The text track
(`squad_text_ml.ipynb`) trains and compares pipelines for question-type
classification on SQuAD.

## Key Results

### Tabular — RandomForest (production model, held-out test set)

Source: `notebooks/artifacts/reports/report_card.json` (stratified 80/20
split, `random_state=42`, 200 test rows).

| Metric | Value |
|---|---|
| ROC-AUC | 0.988 |
| Accuracy (threshold 0.73) | 0.975 |
| Precision | 0.980 |
| Recall | 0.970 |
| F1 | 0.975 |

Top permutation-importance drivers: Daily Internet Usage, Daily Time Spent
on Site, Area Income, Age, Timestamp.

### Image (SVHN) — model comparison

Source: `notebooks/ml_section1_results_224.json`, `ml_section2_pipelines.json`,
`ml_section3_finetune.json`.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Linear SVM (ResNet-18 features, 224px) | 0.628 | 0.612 |
| Logistic Regression (ResNet-18 features, 224px) | 0.625 | 0.613 |
| MLP (ResNet-18 features, 224px) | 0.608 | 0.591 |
| MobileNetV2 features → Logistic Regression | 0.579 | 0.560 |
| ResNet-18, full fine-tune | 0.963 | 0.961 |
| MobileNetV2, full fine-tune | 0.945 | 0.941 |
| ResNet-18, partial fine-tune | 0.899 | 0.890 |

Fine-tuning the CNN backbones substantially outperforms classical models
trained on frozen pretrained features.

## Repository Structure

```
.
├── index.html, tabular.html, tabular_ml.html,
│   text.html, text_ml.html, image.html,
│   image_ml.html, multimodal.html      # Live dashboard pages
├── PRESENTATION_SCRIPT.txt              # Presentation walkthrough script
├── requirements.txt                     # Full environment freeze
├── plotly_data/, plots_data/            # Dashboard chart data
├── sample_images/                       # Sample images for the dashboard
├── reports/                             # Supplementary report assets
└── notebooks/
    ├── STUDY_GUIDE.md                   # End-to-end tabular ML study guide
    ├── requirements.txt                 # Pinned notebook dependencies
    ├── eda_tabular.ipynb
    ├── eda_text.ipynb
    ├── eda_image.ipynb
    ├── eda_multimodal.ipynb
    ├── modeling_tabular.ipynb
    ├── evaluation_tabular.ipynb
    ├── image_ML.ipynb
    ├── squad_text_ml.ipynb
    ├── ml_section1_results_224.json     # Image classifiers, 224px features
    ├── ml_section1_results_32.json      # Image classifiers, 32px features
    ├── ml_section2_pipelines.json       # Image feature/classifier pipelines
    ├── ml_section3_finetune.json        # Image fine-tuning results
    └── artifacts/
        ├── cv_comparison.csv
        ├── model_metadata.json
        ├── random_search_top10.csv
        └── reports/
            ├── report_card.json
            ├── held_out_metrics.csv
            ├── error_slices.csv
            └── permutation_importance.csv
```

## Setup and Run

```bash
pip install -r notebooks/requirements.txt
```

`notebooks/requirements.txt` pins `pandas`, `numpy`, `scikit-learn>=1.3`,
`matplotlib`, `seaborn`, `shap>=0.44`, `joblib`, and `pyarrow`. Open the
notebooks in `notebooks/` (Jupyter or VS Code) and run them in this order for
the tabular track: `eda_tabular.ipynb` → `modeling_tabular.ipynb` →
`evaluation_tabular.ipynb`. The image (`image_ML.ipynb`), text
(`squad_text_ml.ipynb`), and multimodal (`eda_multimodal.ipynb`) notebooks
are independent.
