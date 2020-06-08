# kaggle-hemorrhage-detection
RSNA Intracranial Hemorrhage Detection in kaggle<br>
Leaderboard 1st source

## 1. 데이터 분석(EDA), dicom 추출(png로 변환)
* data_analysis.ipynb

## 2. 데이터 분할 - Train/Valid/Test
* make_dataset.ipynb
* 각각의 train.csv, valid.csv, test.csv로 분리

## 3. CNN 학습
* pytorch에서 제공하는 pretrained densenet121 사용
* 마지막 fc layer이전의 feature들과, 최종 label output 출력
* train.ipynb
* test.ipynb

## 4. 학습된 모델로부터 예측된 label, features 저장
* make_features.ipynb
* dataset폴더에 train_feature.csv, valid_feature.csv, test_feature.csv 생성
* format 
```
filename, label0, label1, ..., label5, feature0, feature1, ..., feature1023
```

## 5. RNN 학습
* train_sequential.ipynb
