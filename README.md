# 컴퓨터 사양
- PatchTST 모델: CPU - Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz, GPU - NVIDIA GeForece RTX 3090(24G) 2개, RAM - 64G

- BigPatchTST 모델: CPU - AMD EPYC 7713 256-Core, GPU - NVIDIA Tesla A100(80G) 8개, RAM - 512G

# 가상환경
- PatchTST 모델: kuka 폴더에서 ```conda env create -f dacon.yaml``` 명령어를 통해 가상환경을 불러옵니다.
- BigPatchTST 모델: PatchTST 폴더의 requirements.txt를 통해 가상환경을 불러옵니다.

# EDA
- eda_product.ipynb 파일을 열고 모든 셀을 실행합니다. dataset 폴더에 features.csv 파일이 생성됩니다.
- eda_timeseries.ipynb 파일을 열고 모든 셀을 실행합니다. dataset 폴더에 total_dates_scaling.csv와 total_dates_scaling_old.csv 두 개의 파일이 생성됩니다.


# Train
Train 과정은 PatchTST, BigPatchTST, 그리고 BigPatchTST++ 모델을 학습하는 것으로 구성되어있습니다. 여기서각 모델들은 파라미터 수가 다른 것으로 PatchTST, BigPatchTST, BigPatchTST++ 순으로 모델의 크기가 큽니다. 

## PatchTST
- PatchTST 폴더에서 patchtst_train.py를 실행시키면 PatchTST/ckpt 폴더에 patchtst_submit_0824.pth 모델 파일이 저장됩니다.
- PatchTST 폴더에서 patchtst_test.py를 실행시키면 모델 추론 결과로 submit 폴더에 patchtst_submit_0824.csv 파일이 생성됩니다.

## BigPatchTST
- PatchTST 폴더에서 bigpatchtst_train1.py를 실행시키면 PatchTST/ckpt 폴더에 bigpatchtst_1.pth 모델 파일이 저장됩니다.
- PatchTST 폴더에서 bigpatchtst_train2.py를 실행시키면 bigpatchtst_1.pth 에 대한추가 학습으로 PatchTST/ckpt 폴더에 bigpatchtst_2.pth 파일이 생성됩니다.
- PatchTST 폴더에서 bigpatchtst_test.py를 실행시키면 bigpatchtst_2.pth 모델 추론 결과로 submit 폴더에 patchtst_submit_0827.csv 파일이 생성됩니다.

## BigPatchTST++
- PatchTST 폴더에서 bigpatchtst++_train.py를 실행시키면 bigpatchtst_2.pth 에 대한 추가 학습으로 PatchTST/ckpt 폴더에 bigpatchtst_3.pth 모델 파일이 저장됩니다.
- PatchTST 폴더에서 bigpatchtst++_test.py를 실행시키면 bigpatchtst_3.pth 모델 추론 결과로 submit 폴더에 patchtst_submit_0828.csv 파일이 생성됩니다.

# Ensemble
Ensemble 과정은 위 Train 과정에서 구한 결과값들을 모두 반영하는 과정입니다. 
- ensemble.ipynb 파일을 열고 모든 셀을 실행합니다.
- ensemble 된 결과값이 submit 폴더에 ensemble_submit_0828_patchtst.csv 파일로 저장됩니다.
- 이 ensemble_submit_0828_patchtst.csv 값이 저희 모델의 최종 값입니다.
