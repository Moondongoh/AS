#<CCTV 영상을 활용한 자동차 추적 model 제작>
# 1. 총 5개의 Class로 이미지 및 라벨파일 생성 (분류 가능한한)
# Class_id -> [0] K3
# Class_id -> [1] Sorento
# Class_id -> [2] Forte
# Class_id -> [3] TheNewGrandStarex
# Class_id -> [4] G90
# ** Yolo를 사용할 경우 객체 좌푯값은 0~1 사이의 값이여야함.
# Yolo11n을 이용해서 모델을 학습
# epoch는 30번씩 batch는 96, 학습률 0.001 설정


import yaml
from ultralytics import YOLO
import os
from sklearn.model_selection import KFold
import shutil
import glob

os.environ["NCCL_P2P_DISABLE"] = "1" 

def prepare_folds(k, dataset_path, label_path):
    # 이미지 파일 경로 불러오기
    image_files = glob.glob(os.path.join(dataset_path, '*.jpg'))
    
    # 레이블 파일 경로 불러오기
    label_files = glob.glob(os.path.join(label_path, '*.txt'))
    
    # KFold 객체 생성
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    return kf.split(image_files), image_files, label_files

def main():
    # 학습 데이터 설정
    data = {
        'train': '',  # 폴드별로 나뉘게 될 경로
        'val': '',    # 폴드별로 나뉘게 될 경로
        'names': ['K3', 'Sorento','Forte', 'G90',
        'TheNewGrandStarex',],
        'nc': 5
    }
    yaml_path = 'D:/New_Tracking/AI_Data.yaml'
    dataset_path = 'D:/train/2024_12_27_Car_detection/New_Class/detect_image'
    label_path = 'D:/train/2024_12_27_Car_detection/New_Class/detect_image2_txt'

    # YAML 파일 생성
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

    # K-폴드 교차 검증 설정 (예: 5-fold)
    k = 5
    fold_indices, image_files, label_files = prepare_folds(k, dataset_path, label_path)

    # K-폴드 교차 검증 수행
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"Training fold {fold + 1}/{k}")

        # 학습 및 검증용 데이터 분할
        train_images = [image_files[i] for i in train_idx]
        train_labels = [label_files[i] for i in train_idx]
        val_images = [image_files[i] for i in val_idx]
        val_labels = [label_files[i] for i in val_idx]

        # 폴드별 학습 및 검증 폴더 생성
        fold_train_dir = f'D:/New_Tracking/fold_{fold}/train'
        fold_val_dir = f'D:/New_Tracking/fold_{fold}/val'
        os.makedirs(os.path.join(fold_train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(fold_train_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(fold_val_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(fold_val_dir, 'labels'), exist_ok=True)

        # 학습 데이터 복사
        for img_file, label_file in zip(train_images, train_labels):
            shutil.copy(img_file, os.path.join(fold_train_dir, 'images'))
            shutil.copy(label_file, os.path.join(fold_train_dir, 'labels'))

        # 검증 데이터 복사
        for img_file, label_file in zip(val_images, val_labels):
            shutil.copy(img_file, os.path.join(fold_val_dir, 'images'))
            shutil.copy(label_file, os.path.join(fold_val_dir, 'labels'))

        # 학습 데이터 YAML 파일 업데이트
        data['train'] = os.path.join(fold_train_dir, 'images')
        data['val'] = os.path.join(fold_val_dir, 'images')

        # YAML 파일 저장
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        # Load a model
        model = YOLO('yolov10n.pt')

        # Train the model on current fold
        model.train(data=yaml_path,
                    epochs=30,
                    imgsz=640,
                    device="0",
                    batch=96,
                    pretrained=False,
                    lr0=0.001,
                    mixup=0.2,   # MixUp 증강 활성화
                    shear=0.1,     # 전단 변환
                    perspective=0.0,  # 투시 변환
                    patience=100,
                    flipud=0.5,
                    mosaic=1.0,
                    translate=0.1,
                    scale=0.5,
                    optimizer='AdamW',
                    save=True,
                    deterministic=False,
                    close_mosaic=10)

if __name__ == '__main__':
    main()