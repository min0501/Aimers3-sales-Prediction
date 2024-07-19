##################################
## 0916에 모델 키움
## MAE 추가
##################################
import random
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from tsai.basics import *

import argparse
from exp.exp_main import Exp_Main
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST

# from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

parser = argparse.ArgumentParser(description="Autoformer & Transformer family for Time Series Forecasting")
# basic config
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument("--train_only", type=bool, default=False, help="perform training on full input dataset without validation and testing")
parser.add_argument("--model_id", type=str, default="test", help="model id")
parser.add_argument("--model", type=str, default="Autoformer",
                    help="model name, options: [Autoformer, Informer, Transformer]")

# forecasting task
parser.add_argument("--seq_len", type=int, default=104, help="input sequence length")
parser.add_argument("--label_len", type=int, default=104, help="start token length")
parser.add_argument("--pred_len", type=int, default=21, help="prediction sequence length")


# DLinear
parser.add_argument("--individual", action="store_true", default=False, help="DLinear: a linear layer for each variate(channel) individually")
# Formers 
parser.add_argument("--embed_type", type=int, default=0, help="0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding")
parser.add_argument("--enc_in", type=int, default=11, help="encoder input size") # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument("--dec_in", type=int, default=11, help="decoder input size")
parser.add_argument("--c_out", type=int, default=11, help="output size")
parser.add_argument("--d_model", type=int, default=64, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=16, help="num of heads")
parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=64, help="num of decoder layers")
parser.add_argument("--d_ff", type=int, default=128, help="dimension of fcn")
parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
parser.add_argument("--factor", type=int, default=1, help="attn factor")
parser.add_argument("--distil", action="store_false",
                    help="whether to use distilling in encoder, using this argument means not using distilling",
                    default=True)
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument("--embed", type=str, default="timeF",
                    help="time features encoding, options:[timeF, fixed, learned]")
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")
parser.add_argument("--do_predict", action="store_true", help="whether to predict unseen future data")

# optimization
parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
parser.add_argument("--itr", type=int, default=2, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="mse", help="loss function")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

# GPU
#parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
#parser.add_argument("--gpu", type=int, default=0, help="gpu")
#parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
#parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
#parser.add_argument("--test_flop", action="store_true", default=False, help="See utils/tools for usage")
args = parser.parse_args('')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CFG = {
    "TRAIN_WINDOW_SIZE": 104,  # 104일치로 학습
    "PREDICT_SIZE": 21,  # 21일치 예측
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 4096,
    "SEED": 41,
    "LAMBDA1": 0.1,
    "LAMBDA2": 10,
    "LR_LAMBDA": 0.85,  # lr scheduler에 사용되는 값
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["SEED"])  # Seed 고정

train_data = pd.read_csv("/root/workspace/data/train.csv").drop(columns=["ID", "제품"])
cond = (train_data["쇼핑몰"] != "S001-00001")
train_data = train_data[cond] #####
output_kuka = train_data.iloc[:, -CFG["PREDICT_SIZE"] :].to_numpy()
product_features = pd.read_csv("/root/workspace/data/features4.csv").drop(
    columns=["Unnamed: 0","ID", "제품", "대분류", "중분류", "소분류", "브랜드","쇼핑몰"]
)
product_features = product_features[cond]
print(train_data)
print(product_features)
time_series_features = pd.read_csv("/root/workspace/data/total_dates_scaling2.csv").drop(
    columns=["Unnamed: 0", "Unnamed: 0.1", "Date"]
)
#time_series_features = time_series_features[cond]
print(time_series_features)

# preprocessing
numeric_cols = train_data.columns[5:]
min_values = train_data[numeric_cols].min(axis=1)
max_values = train_data[numeric_cols].max(axis=1)
ranges = max_values - min_values
ranges[ranges == 0] = 1
train_data[numeric_cols] = (train_data[numeric_cols].subtract(min_values, axis=0)).div(
    ranges, axis=0
)
scale_min_dict = min_values.to_dict()
scale_max_dict = max_values.to_dict()

# Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ["대분류", "중분류", "소분류", "브랜드"]
categorical_nums = [5, 11, 53, 2895]  # 카테고리 별 항목 수

for i, col in enumerate(categorical_columns):
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])
    train_data[col] = train_data[col].div(categorical_nums[i] - 1)  # minmax scaling

# (15890, 1)로 만들기, MinMax Scaling 사용
scaler1 = MinMaxScaler()
scaler1.fit(product_features[["판매량평균"]])
sales_mean_scaled = scaler1.transform(product_features[["판매량평균"]])

scaler2 = MinMaxScaler()
scaler2.fit(product_features[["제품수"]])
prod_num_scaled = scaler2.transform(product_features[["제품수"]])

scaler3 = MinMaxScaler()
scaler3.fit(product_features[["쇼핑몰판매량평균"]])
mall_mean_scaled = scaler3.transform(product_features[["쇼핑몰판매량평균"]])

nprice_mid = product_features[["NormalizedPrice_중분류"]].to_numpy()

product = np.column_stack((sales_mean_scaled, prod_num_scaled, mall_mean_scaled, nprice_mid))

# (459, 1)
sale_info = time_series_features["SaleInfo"].to_numpy()
#holiday = time_series_features["Holiday"].to_numpy()
#salary = time_series_features["Salary"].to_numpy()
holiday_salary = time_series_features["Holiday_Salary"].to_numpy()
month = time_series_features["Month"].to_numpy()
dayofweek = time_series_features["DayofWeek"].to_numpy()

time_series = np.stack((sale_info, holiday_salary, month, dayofweek))  # 필요한 피쳐만 선별


def make_train_data(
    data, train_size=CFG["TRAIN_WINDOW_SIZE"], predict_size=CFG["PREDICT_SIZE"]
):
    """
    학습 기간 블럭, 예측 기간 블럭의 세트로 데이터를 생성
    data : 일별 판매량
    train_size : 학습에 활용할 기간
    predict_size : 추론할 기간
    """
    num_rows = len(data)
    window_size = train_size + predict_size

    input_data = np.empty(
        (
            num_rows * (len(data.columns) - window_size + 1),
            train_size,
            2 + product.shape[1] + time_series.shape[0] + 1,
        )
    )
    target_data = np.empty(
        (
            num_rows * (len(data.columns) - window_size + 1),
            predict_size,
            2 + product.shape[1] + time_series.shape[0] + 1,
        )
    )
    for i in tqdm(range(num_rows), dynamic_ncols=True):
        encode_info_big = np.array(data.iloc[i, 0])  # 대분류 소분류
        encode_info_small = np.array(data.iloc[i, 2])  # 대분류 소분류
        product_data = product[i, :]
        # time_series
        sales_data = np.array(data.iloc[i, 5:])

        for j in range(len(sales_data) - window_size + 1):
            time_series_window = time_series[:, j : j + window_size]
            window = sales_data[j : j + window_size]

            temp_data = np.concatenate(
                (
                    np.tile(encode_info_big, (train_size, 1)),
                    np.tile(encode_info_small, (train_size, 1)),
                    np.tile(product_data, (train_size, 1)),
                    time_series_window[:, :train_size].T,
                    window[:train_size].reshape((-1, 1)),
                ),
                axis=1,
            )
            temp_target = np.concatenate(
                (
                    np.tile(encode_info_big, (predict_size, 1)),
                    np.tile(encode_info_small, (predict_size, 1)),
                    np.tile(product_data, (predict_size, 1)),
                    time_series_window[:, train_size:].T,
                    window[train_size:].reshape((-1, 1)),
                ),
                axis=1,
            )
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = temp_target

    return input_data, target_data


def make_predict_data(data, train_size=CFG["TRAIN_WINDOW_SIZE"]):
    """
    평가 데이터(Test Dataset)를 추론하기 위한 Input 데이터를 생성
    data : 일별 판매량
    train_size : 추론을 위해 필요한 일별 판매량 기간 (= 학습에 활용할 기간)
    """
    num_rows = len(data)

    input_data = np.empty(
        (num_rows, train_size, 2 + product.shape[1] + time_series.shape[0] + 1)
    )

    for i in tqdm(range(num_rows), dynamic_ncols=True):
        encode_info_big = np.array(data.iloc[i, 0])
        encode_info_small = np.array(data.iloc[i, 2])
        product_data = product[i, :]
        sales_data = np.array(data.iloc[i, -train_size:])

        time_series_window = time_series[:, -train_size:]
        window = sales_data[-train_size:]

        temp_data = np.concatenate(
            (
                np.tile(encode_info_big, (train_size, 1)),
                np.tile(encode_info_small, (train_size, 1)),
                np.tile(product_data, (train_size, 1)),
                time_series_window[:, :train_size].T,
                window[:train_size].reshape((-1, 1)),
            ),
            axis=1,
        )
        input_data[i] = temp_data

    return input_data


train_input, train_target = make_train_data(train_data)  # train val total

# test_input = make_predict_data(train_data)  # test(submission)


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])

    def __len__(self):
        return len(self.X)


dataset = CustomDataset(train_input, train_target)

# dataset 분리
data_len = len(dataset)
train_size = int(data_len * 0.8)
val_size = int(data_len * 0.2)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0
)


class PSFALoss(nn.Module):
    """
    PSFA 식과 유사하게 구현한 loss 함수입니다.
    """

    def __init__(self):
        super(PSFALoss, self).__init__()

    def forward(self, inputs, targets):
        share_denominator = torch.sum(targets, axis=0)
        share_denominator[share_denominator == float(0)] = 1  # 나눠지게 만들기
        share = targets / share_denominator
        error_demoninator = torch.max(inputs, targets)
        error_demoninator[error_demoninator == float(0)] = 1  # 나눠지게 만들기
        error = torch.abs(inputs - targets) / error_demoninator
        metric = error * share
        loss = torch.mean(torch.sum(metric, axis=1))
        return loss


def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    mae = nn.L1Loss().to(device)
    psfa = PSFALoss().to(device)
    best_loss = 9999999
    best_model = None

    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        train_loss = []
        # train_mae = []
        for X, Y in tqdm(iter(train_loader), dynamic_ncols=True):
            X = X.to(device)
            Y = Y.to(device)
            Y_sales = Y[:, :, -1]  # dataset의 마지막 컬럼이 판매량
            optimizer.zero_grad()

            output = model(X)
            output_sales = output[:, :, -1]  # dataset의 마지막 컬럼이 판매량
            loss = (
                CFG["LAMBDA1"] * criterion(output, Y)  # mse loss
                + CFG["LAMBDA1"] * mae(output, Y)  # mae loss
                + CFG["LAMBDA2"] * psfa(output_sales, Y_sales)  # psfa loss
            )  # mse, mae loss에 더불어 psfa loss 사용, psfa loss는 판매량에만 적용하였습니다.

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        scheduler.step()  # added

        val_loss = validation(model, val_loader, criterion, mae, psfa, device)
        print(
            f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]"
        )

        if best_loss > val_loss:  # val loss가 가장 작은 모델을 저장
            best_loss = val_loss
            best_model = model
            print("Model Saved")
    return best_model


def validation(model, val_loader, criterion, mae, psfa, device):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader), dynamic_ncols=True):
            X = X.to(device)
            Y = Y.to(device)
            Y_sales = Y[:, :, -1]
            output = model(X)
            output_sales = output[:, :, -1]
            loss = (
                CFG["LAMBDA1"] * criterion(output, Y)
                + CFG["LAMBDA1"] * mae(output, Y)
                + CFG["LAMBDA2"] * psfa(output_sales, Y_sales)
            )

            val_loss.append(loss.item())
    return np.mean(val_loss)


# 모델 선언
model = DLinear.Model(args)
print(torch.cuda.device_count())  # 2개의 GPU를 사용했습니다.
model = nn.DataParallel(model)  # parallel mode
optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer=optimizer, lr_lambda=lambda epoch: CFG["LR_LAMBDA"] ** epoch
)
infer_model = train(model, optimizer, scheduler, train_loader, val_loader, device)

torch.save(infer_model.state_dict(), "/root/workspace/minkyung/ckpt/dlinear_submit_0916_2.pth")
