# 🍽️ Food-101 Image Classification using EfficientNetB0

This project implements an image classification pipeline for the Food-101 dataset using EfficientNetB0 and TensorFlow/Keras.
The model is trained using transfer learning (feature extraction) and achieves ~73% validation accuracy without fine-tuning.

# 📌 Project Overview

Dataset: Food-101 (101 food categories)

Model: EfficientNetB0 (ImageNet pre-trained)

Framework: TensorFlow 2.x

Training Strategy: Feature extraction (base model frozen)

Validation Accuracy: ≈ 73%

Loss Function: Sparse Categorical Crossentropy

# 📂 Dataset Details

Total Classes: 101

Training Images: ~75,750

Validation Images: ~25,250

Source: TensorFlow Datasets (tfds)

The dataset is automatically downloaded and cached using:

tfds.load("food101", split=["train", "validation"])

# 🧠 Model Architecture

The architecture consists of a pretrained EfficientNetB0 backbone followed by a lightweight classification head.

Input (224×224×3)
        ↓
EfficientNetB0 (frozen, ImageNet weights)
        ↓
Global Average Pooling
        ↓
Batch Normalization
        ↓
Dense (101 units, Softmax)

# 🔢 Model Statistics
Parameter Type	Count
Total Params	4,184,072
Trainable Params	131,941
Non-Trainable Params	4,052,131
# 🔄 Data Preprocessing

EfficientNet requires specific preprocessing, which is applied during dataset mapping:

Image resizing to 224 × 224

EfficientNet preprocess_input

Batching & prefetching for performance

image = preprocess_input(image)


# ⚠️ Skipping this step leads to very poor accuracy (~1%)

⚙️ Training Configuration
Setting	Value
Image Size	224 × 224
Batch Size	32
Epochs	10
Optimizer	Adam (lr = 1e-4)
Loss	Sparse Categorical Crossentropy
Callbacks	EarlyStopping, ModelCheckpoint
# 📈 Training Results
Epoch	Train Accuracy	Validation Accuracy
1	23.4%	59.7%
5	66.9%	70.9%
10	72.3%	73.0%

✅ Strong generalization
✅ Stable convergence
✅ No overfitting observed

# 💾 Model Saving

The best performing model is automatically saved during training:

food101_efficientnetb0.keras


This file can be:

Stored in Google Drive

Downloaded locally

Reloaded later for inference or fine-tuning

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install tensorflow tensorflow-datasets

2️⃣ Run the Training Script
python train_food101.py


(The dataset will download automatically on first run.)

🔮 Future Improvements

🔓 Fine-tuning EfficientNet layers

🔁 Data augmentation (RandomFlip, RandomRotation)

📱 Export to TensorFlow Lite (TFLite)

🚀 Deploy as a REST API or mobile app

# ✅ Key Takeaways

EfficientNet requires proper preprocessing

Feature extraction alone can achieve 70%+ accuracy

Transfer learning drastically reduces training time

TensorFlow Datasets simplify large-scale dataset handling

# 📜 License

This project is for educational and research purposes.
Dataset © original Food-101 authors.
