config = {
    "dataset_path": "D:/New_Tracking/Past/detect_image_past",
    "label_path": "D:/New_Tracking/Past/detect_image_past_txt",
    "output_path": "D:/New_Tracking/outputs_Past",
    "yaml_path": "D:/New_Tracking/outputs_Past/data.yaml",
    "model": "yolov8n.pt",
    "k": 5,
    "epochs": 50,
    "imgsz": 640,
    "batch": 32,
    "lr0": 0.001,
    "optimizer": "AdamW",
    "data": {
        "names": ["K3", "Sorento", "Forte", "TheNewGrandStarex", "G90"],
        "nc": 5,
        "train": "",
        "val": ""
    },
    "augmentations": {
        "mixup": 0.2,
        "shear": 0.2,
        "translate": 0.15,
        "scale": 0.5,
        "flipud": 0.4,
    },
}
