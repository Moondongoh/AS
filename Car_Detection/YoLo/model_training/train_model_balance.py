from sklearn.utils import resample
import numpy as np
import os
import shutil
import glob
import yaml
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

def balance_classes(image_files, label_files, labels):
    # 각 클래스의 인덱스별로 데이터를 나누기
    class_indices = {i: [] for i in np.unique(labels)}
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    # 가장 적은 클래스의 샘플 수를 찾기
    min_class_size = min(len(indices) for indices in class_indices.values())
    
    # 각 클래스에서 최소 샘플 수만큼만 샘플링 (언더샘플링)
    balanced_indices = []
    for class_id, indices in class_indices.items():
        class_sample = np.random.choice(indices, min_class_size, replace=False)
        balanced_indices.extend(class_sample)

    # 균등하게 샘플링된 이미지와 레이블 리스트 반환
    balanced_image_files = [image_files[i] for i in balanced_indices]
    balanced_label_files = [label_files[i] for i in balanced_indices]
    balanced_labels = labels[balanced_indices]
    
    return balanced_image_files, balanced_label_files, balanced_labels

def prepare_folds(k, dataset_path, label_path):
    # 이미지와 레이블 파일을 읽어옵니다.
    image_files = glob.glob(os.path.join(dataset_path, '*.jpg'))
    label_files = glob.glob(os.path.join(label_path, '*.txt'))

    # 각 레이블 파일에서 클래스 정보 추출 (각 레이블 파일의 첫 번째 값이 클래스)
    labels = []
    for label_file in label_files:
        with open(label_file, 'r') as f:
            # 첫 번째 줄의 첫 번째 값을 클래스 번호로 가정
            label = int(f.readline().split()[0])
            labels.append(label)

    labels = np.array(labels)

    # 클래스 균형 맞추기
    image_files, label_files, labels = balance_classes(image_files, label_files, labels)

    # StratifiedKFold를 사용하여 계층적 샘플링을 통해 데이터를 나눕니다.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    return skf.split(image_files, labels), image_files, label_files

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
