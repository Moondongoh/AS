import os
from typing import List, Tuple, Set

def extract_unique_models(directory: str) -> Set[str]:
    """디렉토리 내 모든 txt 파일에서 고유한 차종 모델명을 추출합니다."""
    models = set()
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            if len(parts) >= 2:
                models.add(parts[1])
    
    return models

def create_model_mapping(models: Set[str]) -> dict:
    """사용자와 대화형으로 각 모델의 새로운 이름을 입력받습니다."""
    model_mapping = {}
    
    print("\n=== 차종별 새 이름 입력 ===")
    print("(변경하지 않으려면 엔터를 누르세요)")
    
    for model in sorted(models):  # 정렬해서 보기 좋게 표시
        while True:
            new_name = input(f"'{model}' 의 새로운 이름: ").strip()
            if new_name == "":  # 엔터만 누르면 변경하지 않음
                break
            if '_' in new_name:  # 파일명 형식 유지를 위해 '_' 불가
                print("새 이름에는 '_'를 포함할 수 없습니다.")
                continue
            model_mapping[model] = new_name
            break
    
    return model_mapping

def bulk_rename_files(directory: str, model_mapping: dict) -> Tuple[List[str], List[str]]:
    """파일명 변경을 실행합니다."""
    success_files = []
    failed_files = []
    
    # 변경 전에 미리보기 제공
    preview = []
    name_count = {}  # 각 모델명별 카운터
    
    # 먼저 모든 파일을 수집하고 정렬
    files_to_process = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            if len(parts) >= 2 and parts[1] in model_mapping:
                files_to_process.append(filename)
    
    # 파일명 정렬 (선택사항)
    files_to_process.sort()
    
    # 각 파일에 대해 새 이름 생성
    for filename in files_to_process:
        parts = filename.split('_')
        if len(parts) >= 2 and parts[1] in model_mapping:
            base_name = model_mapping[parts[1]]
            
            # 카운터 증가
            name_count[base_name] = name_count.get(base_name, 0) + 1
            
            # 첫 번째 파일은 번호 없이, 그 이후부터는 (i) 형식으로
            if name_count[base_name] == 1:
                new_filename = f"{base_name}.txt"
            else:
                new_filename = f"{base_name}({name_count[base_name]-1}).txt"
            
            preview.append((filename, new_filename))
    
    # 미리보기 출력
    print("\n=== 변경 예정 파일 목록 ===")
    for old, new in preview:
        print(f"{old} -> {new}")
    
    # 사용자 확인
    confirm = input("\n이대로 진행할까요? (y/n): ").lower().strip()
    if confirm != 'y':
        print("작업이 취소되었습니다.")
        return [], []
    
    # 실제 파일명 변경 실행
    for old_name, new_name in preview:
        try:
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            success_files.append(f"{old_name} -> {new_name}")
        except Exception as e:
            failed_files.append(f"{old_name} (에러: {str(e)})")
    
    return success_files, failed_files

def print_results(success_files: List[str], failed_files: List[str]) -> None:
    """결과를 출력합니다."""
    print("\n=== 성공한 파일 ===")
    for file in success_files:
        print(f"✓ {file}")
        
    if failed_files:
        print("\n=== 실패한 파일 ===")
        for file in failed_files:
            print(f"✗ {file}")
    
    print(f"\n총 처리: {len(success_files)} 성공, {len(failed_files)} 실패")

# 메인 실행 코드
if __name__ == "__main__":
    # 파일들이 있는 디렉토리 경로 설정
    directory_path = "D:/Car_Detection/txt_result"
    if not directory_path:
        directory_path = "."
    
    # 존재하는 모델명 추출
    models = extract_unique_models(directory_path)
    print(f"\n발견된 차종: {', '.join(sorted(models))}")
    
    # 사용자로부터 새 이름 입력받기
    model_mapping = create_model_mapping(models)
    
    # 변경할 내용이 있는 경우에만 진행
    if model_mapping:
        # 파일명 변경 실행
        successful, failed = bulk_rename_files(directory_path, model_mapping)
        # 결과 출력
        print_results(successful, failed)
    else:
        print("\n변경할 내용이 없어 작업을 종료합니다.")