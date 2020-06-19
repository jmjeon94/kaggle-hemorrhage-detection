# kaggle-hemorrhage-detection
RSNA Intracranial Hemorrhage Detection in kaggle<br>
Leaderboard 1st source

## 0. 데이터셋 가공
* 소스: jupyter/make_dataset.ipynb
* kaggle 제공 dataframe -> label, meta 정보가 추가된 dataframe

## 1. 데이터 분석(EDA), dicom 추출(png로 변환)
* 소스: jupyter/EDA_stage_1.ipynb, jupyter/data_analysis.ipynb
* dicom 추출은 2가지 방법
```
1. 하나의 slice기준으로 n-1, n, n+1을 concatenate한다.
2. 하나의 slice를 3개의 window level, width로 windowing하여 concatentate한다.
```

## 2. 데이터 분할 - Train/Valid/Test
* 소스: jupyter/split_dataset.ipynb
* 각각의 train.csv, valid.csv, test.csv로 분리
* hemorrhage, normal 환자를 각각 500명, 100명 추출

| | 사람 수 | slice 수 |
|:---|:---:|:---:|
| Train | 480명 | 16663장 |
| Valid | 60명 | 2051장 |
| Test | 60명 | 2073장 |


## 3. CNN 학습
* 소스: train.py, test.py
* config파일: /config/config_cnn.yaml 참조하여 수정
* pytorch에서 제공하는 pretrained densenet121 사용
* 마지막 fc layer이전의 feature들과, 최종 label output 출력
* checkpoint는 timestamp로 /checkpoints/cnn에 저장
* tensorboards는 /tensorboard/cnn에 저장
```python
$ python3 train.py
$ python3 test.py
```

## 4. 학습된 모델로부터 예측된 label, features 저장
* 소스: make_features.ipynb
* dataset폴더에 train_feature.csv, valid_feature.csv, test_feature.csv 생성
* feature 파일 columns
```
filename, label0, label1, ..., label5, feature0, feature1, ..., feature1023
```

## 5. RNN 학습
* 소스: train_sequential.py, test_sequential.py
* config파일: /config/config_rnn.yaml 참조하여 수정
* 위 feature파일과 output label이 input이 됨
* checkpoint는 timestamp로 /checkpoints/cnn에 저장
* tensorboards는 /tensorboard/cnn에 저장
```python
$ python3 train_sequential.py
$ python3 test_sequential.py
```

## Architecture


