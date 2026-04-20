"""
CIFAR-100은 100개 fine class를 20개의 super-class(coarse class)로 묶은 2-level 계층 구조를 갖는다.
아래 `CIFAR100_FINE_TO_COARSE[i]` 는 fine class i가 속한 super-class(0..19) 인덱스다.
이 매핑은 CIFAR-100 원본 meta 파일의 공식 순서로, 아래와 같다.

Super-class grouping
  0  aquatic_mammals          1  fish                 2  flowers
  3  food_containers          4  fruit_and_vegetables 5  household_electrical
  6  household_furniture      7  insects              8  large_carnivores
  9  large_man-made_outdoor   10 large_natural_outdoor 11 large_omnivores
  12 medium_mammals           13 non-insect_invertebrates 14 people
  15 reptiles                 16 small_mammals        17 trees
  18 vehicles_1               19 vehicles_2

평가 지표 "Super-Class Accuracy" 는 Top-5 예측 중 GT 와 같은 super-class 에 속한 비율이다.
"""

# fine class 0..99 각각이 속한 super-class (20개) 인덱스
CIFAR100_FINE_TO_COARSE = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,   # 0-9
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,   # 10-19
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,   # 20-29
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,   # 30-39
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,   # 40-49
    16,  4, 17,  4,  2,  0, 17,  4, 18, 17,  # 50-59
    10,  3,  2, 12, 12, 16, 12,  1,  9, 19,  # 60-69
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,   # 70-79
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,  # 80-89
    18,  1,  2, 15,  6,  0, 17,  8, 14, 13,  # 90-99
]

# Sanity Check
assert len(CIFAR100_FINE_TO_COARSE) == 100
assert set(CIFAR100_FINE_TO_COARSE) == set(range(20))

# ImageNet 통계 (DHVT 원본을 따라 CIFAR 학습에도 ImageNet 정규화 사용)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)