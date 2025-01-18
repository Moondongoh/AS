import yaml
import os
import shutil
import glob
from sklearn.model_selection import KFold
from ultralytics import YOLO

def prepare_folds(k, dataset_path, label_path):
    image_files = glob.glob(os.path.join(dataset_path, '*.jpg'))
    label_files = glob.glob(os.path.join(label_path, '*.txt'))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return kf.split(image_files), image_files, label_files

def train_model(config):
    # YAML 파일의 디렉토리 경로를 확인하고 생성
    yaml_dir = os.path.dirname(config["yaml_path"])
    os.makedirs(yaml_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # YAML 파일 생성
    with open(config["yaml_path"], 'w') as f:
        yaml.dump(config["data"], f)

    # K-폴드 설정
    k = config["k"]
    fold_indices, image_files, label_files = prepare_folds(k, config["dataset_path"], config["label_path"])

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"Training fold {fold + 1}/{k}")

        # 데이터 분할
        train_images = [image_files[i] for i in train_idx]
        train_labels = [label_files[i] for i in train_idx]
        val_images = [image_files[i] for i in val_idx]
        val_labels = [label_files[i] for i in val_idx]

        # 폴드별 폴더 생성
        fold_train_dir = os.path.join(config["output_path"], f'fold_{fold}', 'train')
        fold_val_dir = os.path.join(config["output_path"], f'fold_{fold}', 'val')
        os.makedirs(os.path.join(fold_train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(fold_train_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(fold_val_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(fold_val_dir, 'labels'), exist_ok=True)

        # 데이터 복사
        for img, lbl in zip(train_images, train_labels):
            shutil.copy(img, os.path.join(fold_train_dir, 'images'))
            shutil.copy(lbl, os.path.join(fold_train_dir, 'labels'))
        for img, lbl in zip(val_images, val_labels):
            shutil.copy(img, os.path.join(fold_val_dir, 'images'))
            shutil.copy(lbl, os.path.join(fold_val_dir, 'labels'))

        # YAML 경로 업데이트
        config["data"]["train"] = os.path.join(fold_train_dir, 'images')
        config["data"]["val"] = os.path.join(fold_val_dir, 'images')

        # YAML 저장
        with open(config["yaml_path"], 'w') as f:
            yaml.dump(config["data"], f)

        # 모델 학습
        model = YOLO(config["model"])
        model.train(
            data=config["yaml_path"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            lr0=config["lr0"],
            optimizer=config["optimizer"],
            **config["augmentations"]
        )
