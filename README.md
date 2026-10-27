# 🩺 SonoTriage

**AI that triages breast ultrasounds, explained.**

SonoTriage is a free decision-support tool that reads a breast ultrasound image and flags it as **normal**, **benign-looking** or **malignant-looking**, with a confidence score for every class. It was built for the **2026 International CodeTheCure Hackathon** (category: *Detection & Diagnosis*).

> ⚠️ **Not a diagnosis.** SonoTriage is a screening-support prototype for education and research. It does not replace a doctor, and a confirmed diagnosis requires clinical examination and biopsy.

---

## Why this project

Breast cancer outcomes depend heavily on early detection, but radiologists are scarce in many low-resource regions. A lightweight, free tool that helps prioritize which ultrasound images should be reviewed first by a clinician can make screening more accessible.

## Features

- Upload a breast ultrasound image (PNG / JPG) and get a result in seconds
- Three-class output: **normal**, **benign**, **malignant**
- Confidence for every class, with a low-confidence warning
- Clean web interface built with Streamlit
- Runs on CPU, so it needs no special hardware

## Dataset

[BUSI (Breast Ultrasound Images Dataset)](https://doi.org/10.1016/j.dib.2019.104863): 780 ultrasound images from 600 women, in three classes.

| Class     | Images |
| --------- | ------ |
| Benign    | 437    |
| Malignant | 210    |
| Normal    | 133    |

Segmentation mask files were removed; only the original images are used for classification.

> Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *Dataset of breast ultrasound images.* Data in Brief, 2020; 28:104863.

## Method

**Model:** EfficientNet-B0 pretrained on ImageNet, with a custom classification head (1280 → 256 → 3, dropout 0.3).

**Training pipeline**

1. Stratified split into train / validation / test (70 / 15 / 15)
2. Images resized to 224 × 224 and normalized with ImageNet mean and std
3. Light augmentation: horizontal flip, rotation (10°), brightness and contrast jitter
4. Class-weighted loss with label smoothing (0.1) to handle class imbalance
5. **Phase 1, transfer learning:** backbone frozen, only the new head trained (lr 1e-3, 15 epochs)
6. **Phase 2, fine-tuning:** last blocks of the backbone unfrozen (lr 1e-4, cosine schedule, 25 epochs)
7. Best checkpoint chosen by **validation loss**; the test set was used only for the final evaluation

Fine-tuning the last 3 and the last 5 blocks gave very similar validation loss. Test-time augmentation (horizontal flip) was also tried and gave no improvement, so it is not used.

## Results

Evaluated once on a held-out test set of 117 images.

| Metric            | Value            |
| ----------------- | ---------------- |
| Accuracy          | 82.9%            |
| AUC (one-vs-rest) | 0.938            |
| Malignant recall  | 87.1% (27 of 31) |

| Class     | Precision | Recall | F1    | Support |
| --------- | --------- | ------ | ----- | ------- |
| Benign    | 0.911     | 0.773  | 0.836 | 66      |
| Malignant | 0.692     | 0.871  | 0.771 | 31      |
| Normal    | 0.864     | 0.950  | 0.905 | 20      |

**Confusion matrix** (rows: true class, columns: predicted)

|                     | Benign | Malignant | Normal |
| ------------------- | ------ | --------- | ------ |
| **Benign**    | 51     | 12        | 3      |
| **Malignant** | 4      | 27        | 0      |
| **Normal**    | 1      | 0         | 19     |

For a screening tool, missing a malignant case is the most costly error, so malignant recall is the key metric. The trade-off is that some benign images (12 of 66) are flagged as malignant-looking.

## Run it locally

```bash
git clone https://github.com/tahanadeem112299/cnn_cancer_assistant.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal and upload an ultrasound image.

## Project structure

```
├── app.py                 # Streamlit web app
├── busi_model.pt          # Trained model weights
├── requirements.txt       # Python dependencies
├── breast_cancer.ipynb    # Training and evaluation notebook
└── README.md
```

## Limitations

Please read these before relying on any result:

- **Small dataset:** about 780 images from a single source, so the model may not generalize to other scanners, hospitals or populations.
- **Small test set:** only 117 test images (31 malignant), so the metrics have wide uncertainty. For example, malignant recall of 87% could reasonably be anywhere from about 70% to 95%.
- **No patient-level split:** BUSI has no patient IDs, so images from the same patient may appear in both train and test sets, which can make results look better than they are.
- **No clinical validation:** the model has not been tested in a real clinical workflow.
- **Not a diagnosis:** outputs are probabilities from a model, not medical findings.

## Future work

- Grad-CAM heatmaps in the app to show which region influenced the prediction
- Adjustable decision threshold to trade off false alarms against missed malignant cases
- Ensemble of several models (for example EfficientNet and ResNet)
- Validation on additional public and clinical datasets
- Patient-level data splits and cross-validation

## Tech stack

Python · PyTorch · torchvision · Streamlit · scikit-learn · Google Colab

## Acknowledgements

- BUSI dataset authors (Al-Dhabyani et al.)
- CodeTheCure Hackathon organizers and mentors
