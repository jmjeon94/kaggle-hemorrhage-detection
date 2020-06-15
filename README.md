# kaggle-hemorrhage-detection
RSNA Intracranial Hemorrhage Detection in kaggle<br>
Leaderboard 1st source

## 0. 데이터셋 가공
* 소스: make_dataset.ipynb
* kaggle 제공 dataframe -> label, meta가 추가된 dataframe

## 1. 데이터 분석(EDA), dicom 추출(png로 변환)
* 소스: EDA_stage_1.ipynb, data_analysis.ipynb
* dicom 추출은 2가지 방법
```
1. 하나의 slice기준으로 n-1, n, n+1을 concatenate한다.
2. 하나의 slice를 3개의 window level, width로 windowing하여 concatentate한다.
```

## 2. 데이터 분할 - Train/Valid/Test
* 소스: split_dataset.ipynb
* 각각의 train.csv, valid.csv, test.csv로 분리

## 3. CNN 학습
* 소스: train.ipynb
* pytorch에서 제공하는 pretrained densenet121 사용
* 마지막 fc layer이전의 feature들과, 최종 label output 출력
* checkpoint는 timestamp로 /checkpoints/cnn에 저장
* tensorboards는 /tensorboard/cnn에 저장

## 4. 학습된 모델로부터 예측된 label, features 저장
* 소스: make_features.ipynb
* dataset폴더에 train_feature.csv, valid_feature.csv, test_feature.csv 생성
* feature 파일 columns
```
filename, label0, label1, ..., label5, feature0, feature1, ..., feature1023
```

## 5. RNN 학습
* 소스: train_sequential.ipynb
* 위 feature파일이 input이 됨
* checkpoint는 timestamp로 /checkpoints/cnn에 저장
* tensorboards는 /tensorboard/cnn에 저장

