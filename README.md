# 🩻 Chest X-Ray Pathology Classification — Grand X-Ray Slam (Division B)

This repository contains my solutions for the **[Grand X-Ray Slam (Division B)](https://www.kaggle.com/competitions/grand-xray-slam-division-b)** Kaggle competition.  
The objective is to develop a deep learning model that can **automatically classify chest X-rays into multiple pathology classes** with high diagnostic accuracy.

---

## 🚀 Project Overview

- Processed **40K+ chest X-ray images across 14 pathology classes** to build a high-quality dataset.  
- Applied **CLAHE-based contrast enhancement**, removed corrupted images, and performed **advanced data augmentation** (rotations, flips, random crops) for better generalization.  
- Trained a **DenseNet-121 multi-label classifier** with **5-fold cross-validation**, achieving an average **macro-AUC of 0.91 ± 0.02** using `BCEWithLogitsLoss` and ROC-AUC metrics.  
- Experimented with a **Qwen Vision Transformer pipeline** for multimodal (vision-language) learning and improved robustness.  

---

## 🧠 Models Implemented

| Model | Description | Framework |
|--------|--------------|------------|
| **DenseNet-121** | CNN-based baseline pretrained on ImageNet and fine-tuned for 14-class multi-label classification. | PyTorch |
| **Qwen Vision Pipeline** | Multimodal transformer using image + language embeddings for X-ray reasoning. | Hugging Face Transformers |

---

## 📁 Repository Structure

```
.
├── densenet-1.ipynb          # DenseNet-121 training & evaluation notebook
├── Qwen Pipeline.ipynb       # Qwen Vision multimodal pipeline
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies (optional)
```

---

## 🏋️ Model Training

### DenseNet-121 Training
```python
# Inside densenet-1.ipynb
# - Load dataset and preprocess with CLAHE
# - Apply data augmentation
# - Train with BCEWithLogitsLoss and 5-fold CV
# - Evaluate macro-AUC and save best weights
```

### Qwen Vision Pipeline
```python
# Inside Qwen Pipeline.ipynb
# - Use Qwen-Vision model for multimodal X-ray understanding
# - Optionally fine-tune on annotated text prompts
# - Evaluate zero-shot or fine-tuned performance
```

---

## 📊 Results Summary

| Model | Notes |
|--------|-------|
| DenseNet-121 | Strong CNN baseline |
| Qwen Vision Pipeline | Multimodal experimentation |

---

## 📈 Visualizations

- **Training curves** for loss and ROC-AUC  
- **Grad-CAM heatmaps** for explainability  
- **Confusion matrices** to identify misclassifications  

(Plots are available inside the notebooks.)

---

## 🧩 Future Enhancements

- Incorporate **ensemble averaging** between DenseNet and Qwen models  
- Use **Swin Transformer** and **EfficientNet-V2** for stronger baselines  
- Integrate **explainability metrics** and uncertainty estimation  

---

## 🏁 Acknowledgments

- [Kaggle — Grand X-Ray Slam Division B](https://www.kaggle.com/competitions/grand-xray-slam-division-b)  
- [PyTorch](https://pytorch.org/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Qwen-VL](https://huggingface.co/Qwen)

---

## 📜 License

This repository is licensed under the **MIT License**.  
Feel free to use, modify, and share with attribution.

---
