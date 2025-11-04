# Object-Counting-with-Yolov8
For this project, we tested with **Python 3.10.18** and **NVIDA GPU (CUDA 12.1)** (optional but recommended for training and inference). Use ffmpeg to extract video frame for polygonzone.

## 1. Environment setup
### 1.1 Conda
For anaconda3 user, please use the following:
```
# Create and activate
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
```
Install PyTorch for CUDA 12.1
```
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU-only alternative:
# conda install pytorch torchvision cpuonly -c pytorch
```
Dependencies
```
pip install -r requirements.txt
```

### 1.2 Python venv:
If you are not using anaconda3, you can consider using Python virtual environment:
```
# Create & activate
python -m venv <env_name>
# Windows:
<env_name>\Scripts\activate
# macOS/Linux:
source <env_name>/bin/activate
```

Install PyTorch for CUDA 12.1
```
# CUDA 12.1 wheels:
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU-only alternative:
# pip install torch torchvision
```

## 2. Code summary
`download_roboflow_dataset.py` — pull the apples dataset from Roboflow.

`config.py` — define global config for training

`train.py` — finetune pretrained YOLO model from COCO checkpoint

`evaluate.py` — reads results.csv from a training run and produces PNG plots (loss, mAP, P/R, LR).

`utils/args.py` — Ultralytics train kwargs and CLI argument parsing.

`Augmentation_and_HyperParameters.py` — helpers for augmentations & hyperparameters 

`logging.py` — Use Comet ML for logging training session

## 3. Download the dataset
For the dataset used for training, we used the one from Roboflow https://universe.roboflow.com/test-wlzsc/apple-detection-jnzwb/
To be able to download the dataset, first input your API key and YOLO's version to download in `download_roboflow_dataset.py`
Then:
```python download_roboflow_dataset.py```

## 4. Training:
```
python train.py \
  --data path/to/data.yaml \
  --model <load_from_pretrained_YOLO_model> \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device auto \
  --project runs/detect \
  --name <input_trained_model_name> \
  --logger comet
```

## 5. Evaluation and Plotting
```
python evaluate.py \
  --csv runs/detect/<your_trained_model_name>/results.csv \
  --out plots
```

## 6. Inference
Fill in your input video at trained model at `main.py`. After that, use ```python main.py``` for inference
