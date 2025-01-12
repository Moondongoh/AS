config = {
    "dataset_path": "D:/New_Tracking/Present/detect_image_present",
    "label_path": "D:/New_Tracking/Present/detect_image_present_txt",
    "output_path": "D:/New_Tracking/outputs_Present",
    "yaml_path": "D:/New_Tracking/outputs_Present/data.yaml",
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
        "shear": 0.1,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.5,
    },
}
