# 🍌 Banana Classification with YOLOv11-OBB

This project uses a pretrained **YOLOv11-OBB (Oriented Bounding Box)** model to classify and localize bananas in an image. It contains:

- `testmodel_1.pt` — Custom trained YOLOv11 model  
- `test.py` — Script to perform detection  
- `https://drive.google.com/drive/folders/1CH_rKAKSQB37OHolwD_8TwWugtojEgyV?usp=drive_link` — Dataset used for training (optional for inference)

---

## 🛠️ Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/MingxS-cs/Yolov11OBB.git
cd Yolov11OBB
```

### 2. Create and activate Conda environment

Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

```bash
conda env create -f environment.yml
conda activate ATEC
```

This will install all required dependencies including:
- YOLOv11-OBB
- Ultralytics
- MMEngine, MMDetection, PyTorch, OpenCV, etc.

---

---

## 🚀 How to Run

### 1. Prepare test image

Place your test image (e.g., `test1.jpg`) in the project folder.

### 2. Run detection

```bash
python test.py
```

You will see a pop-up window showing the result, and an output image saved in the `runs` folder.

---

## 🧠 Model Info

- `testmodel_1.pt`: A YOLOv11s-OBB model trained on a banana classification dataset.
- It uses oriented bounding boxes (rotated rectangles) to more accurately localize banana objects.

---

## 🧪 test.py Overview

```python
from ultralytics import YOLO

model = YOLO("testmodel_1.pt")  # Load the custom model
results = model.predict("test1.jpg", show=True, save=True)  # Run inference

# Access results:
for result in results:
    xywhr = result.obb.xywhr
    xyxyxyxy = result.obb.xyxyxyxy
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]
    confs = result.obb.conf
    print(f"xywhr: {xywhr}, xyxyxyxy: {xyxyxyxy}, names: {names}, confs: {confs}")
```

---

## 🧩 Dataset

The `dataset` file in `https://drive.google.com/drive/folders/1CH_rKAKSQB37OHolwD_8TwWugtojEgyV?usp=drive_link` contains the training dataset and label information.  
It is optional for using the pretrained model but required if you plan to **retrain or fine-tune**.

---



## 📜 License

This project is for research and educational purposes only.
