import os
from ultralytics import YOLO

def calculate_accuracy(model_path, test_path, class_names):
    # 모델 로드
    model = YOLO(model_path)

    # 테스트 이미지 파일 불러오기
    test_images = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 클래스별 매칭 정보
    total = 0
    correct = 0

    # 테스트 이미지 예측
    for img_path in test_images:
        # 이미지 이름에서 확장자를 제외하고 클래스 추출 (예: K3_1.jpg -> K3)
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]  # 확장자 제거
        true_class = img_name_no_ext.split('_')[0]       # 언더바 이전의 단어 추출

        # 예측 수행
        results = model(img_path)  # 예측 수행
        predictions = results[0].boxes.cls  # 예측된 클래스 IDs
        confidences = results[0].boxes.conf  # 예측된 클래스 확률
        
        # 기본값 설정 (예측이 없는 경우 대비)
        predicted_class = "No prediction"
        
        if len(predictions) > 0:
            # 가장 높은 확률을 가진 클래스 ID
            top_prediction = int(predictions[0].item())
            predicted_class = class_names[top_prediction]

            # 정답 비교
            if true_class == predicted_class:
                correct += 1

        # 모든 이미지에 대해 total 증가
        total += 1

        print(f"Image: {img_name}, True: {true_class}, Predicted: {predicted_class}")

    # 정확도 계산
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    model_path = "D:/New_Tracking/runs/detect/train25/weights/best.pt"  # <<<<<< ******train 번호 수정 필요함****** 16~20 //21~25<<<<< 17이 갯수 ㅇ 25는 x

    test_path = "D:/New_Tracking/Test" 
    #D:/New_Tracking/other_Test/Processing    //학습에 미사용 사진
    #D:/New_Tracking/Test                     //학습에 사용 사진
    class_names = ["K3", "Sorento", "Forte", "TheNewGrandStarex", "G90"] 

    calculate_accuracy(model_path, test_path, class_names)
