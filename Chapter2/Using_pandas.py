import os
import pandas as pd

import torch


os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


data = pd.read_csv(data_file)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

inputs = inputs.fillna(inputs.mean(numeric_only=1))

inputs = pd.get_dummies(inputs, dummy_na=1)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(X.T)
