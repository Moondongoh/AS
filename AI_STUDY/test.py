import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 입력 이미지 (7x7)
image = torch.tensor([
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 7, 7)

# 2. 필터 정의 (3x3)
filters = torch.tensor([
    [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 필터 1
    [[0, 0, 0], [1, 1, 1], [0, 0, 0]],  # 필터 2
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]]   # 필터 3
], dtype=torch.float32).unsqueeze(1)  # (3, 1, 3, 3)

# 3. CNN 레이어 정의 및 필터 적용
conv = F.conv2d(image, filters)

# 4. 결과 출력 및 맥스풀링 적용
print("Convolution Result:")
print(conv)

# 맥스풀링 적용 (2x2 커널, 스트라이드 2)
max_pool = F.max_pool2d(conv, kernel_size=2, stride=2)
print("\nMax Pooling Result:")
print(max_pool)
