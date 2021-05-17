#install numpy
#install pandas
#install pytorch
#install datetime(自己下載whl黨)
# You should not modify this part.
import torch
import torch.nn as nn
import pandas as pd
import numpy
from datetime import datetime, timedelta
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

#模型架構(LSTM)


class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):  # 2,32,2,24
                super(LSTM, self).__init__()
                self.hidden_dim = hidden_dim  # hidden_dim=hidden_layer的output dim(也是1個hidden layer的 LSTM neuron個數)
                self.num_layers = num_layers  # 幾層hidden layer(不能太多!!)
                self.device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)#輸出尺寸是(batch,hidden_dim)
                self.fc = nn.Linear(hidden_dim, output_dim)  # ouput_dim 是指最後要predict的日子是幾天(1!!)
                self.fc1=nn.Linear(output_dim,2)
                self.relu=nn.ReLU(inplace=True)

        def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
                #分 generation consumption輸出
                out1 = self.relu(self.fc(hn[1]))#for generation #out size=(batch size,len_seq,hidden_dim)
                out2 = self.fc1(out1)#for consumption   # out size=(batch size,len_seq,hidden_dim)
                return out2


# #data loader(input size[batch,any,2] output=[batch,1,2])
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# class TestDataset(Dataset):
#     def __init__(self,gen_path,con_path):
#         self.gen_path=gen_path #args.generation
#         self.con_path=con_path# args.consumption
#         self.n_samples=1 #7*24=168筆資料
#         # read csv
#         generation = pd.read_csv(self.gen_path)
#         consumption = pd.read_csv(self.con_path)
#         self.re_x=[]
#         for i in range(len(generation['time'])):
#             temp=[]
#             temp.append(generation[i])
#             temp.append(consumption[i])
#             self.re_x.append(temp)
#         self.re_x=[re_x]
#     def __getitem__(self, index):
#         """ Changing list to tensor. """
#         list_of_tensors = [torch.tensor(np.array(i)) for i in self.re_x[index]]  # self.re_x[index]為list [[gen,con]*168]
#         self.re_x[index] = torch.stack(list_of_tensors)
#         # 168*2大小
#
#         return self.re_x[index]
#
#     def __len__(self):
#         return self.n_samples  # 1


if __name__ == "__main__":
    c_mean = 1.443836
    c_std = 1.630945
    g_mean = 0.780310597
    g_std = 1.413282181

    args = config()
    #讀取模型+參數黨
    input_dim = 2  # 放入 [generation,consumption] 做training #input_dim是指輸入維度
    hidden_dim = 128  # 代表一層hidden layer有128個LSTM neuron
    num_layers = 2  # 2層hidden layer
    output_dim = 64  # 最後要predict的日子是幾天(predict 1 day 的 open price)#原:24
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    model = model.float()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('params.pkl'))
    else:
        model.load_state_dict(torch.load('params.pkl',map_location='cpu'))
    #放入input data
    # read csv
    generation = pd.read_csv(args.generation)
    consumption = pd.read_csv(args.consumption)
    input_x = []#168*2
    for i in range(len(generation['time'])):
        temp = []
        temp.append((generation['generation'][i]-g_mean)/g_std)
        temp.append((consumption['consumption'][i]-c_mean)/c_std)
        input_x.append(temp)
    #轉tensor
    tensor_input_x=torch.tensor(input_x[0]).expand(1,2)
    for i in range(1,len(input_x)):
        tensor_input_x=torch.cat((tensor_input_x,torch.tensor(input_x[i]).expand(1,2)),dim=0)
    tensor_input_x=tensor_input_x.unsqueeze(0)#[1,168,2]
    print(tensor_input_x.shape)

    #針對output做決策(data)
    data=[]
    hour=generation['time'][167]#2018-08-31 23:00:00
    tem=hour.split(' ')
    tt=tem[0].split('-')+tem[1].split(':')
    now=datetime(int(tt[0]), int(tt[1]), int(tt[2]),int(tt[3]) , int(tt[4]), int(tt[5]))

    model.eval()
    for i in range(24):
        y_pred= model(tensor_input_x[:,i:,:].to(device).float())
        y_pred=y_pred.squeeze(0)
        now = now + timedelta(hours=1)
        next_hour = now.strftime("%Y-%m-%d %H:%M:%S")
        if y_pred[0]<0:
            y_pred[0]=0
        if y_pred[1]<0:
            y_pred[1]=0
        #if y_pred[0]==y_pred[1]:#do nothing
        if y_pred[0]>y_pred[1]: #代表有餘電要賣
            data.append([next_hour,"sell",2.48,round((y_pred[0]-y_pred[1]).tolist()*0.1,2)])
            data.append([next_hour,"sell",2.35,round((y_pred[0]-y_pred[1]).tolist()*0.2,2)])
            data.append([next_hour, "sell", 2.2,round((y_pred[0]-y_pred[1]).tolist()*0.7,2)])
            data.append([next_hour, "sell", 2.0,round((y_pred[0]-y_pred[1]).tolist()*0.1,2)])

        elif y_pred[0] < y_pred[1]: #代表電不足 要買
            data.append([next_hour, "buy", 2.51, round((y_pred[1] - y_pred[0]).tolist() * 0.7, 2)])
            data.append([next_hour, "buy", 2.4, round((y_pred[1] - y_pred[0]).tolist() * 0.2, 2)])
            data.append([next_hour, "buy", 2.25, round((y_pred[1] - y_pred[0]).tolist() * 0.1, 2)])
            data.append([next_hour, "buy", 2.1,round((y_pred[1] - y_pred[0]).tolist() * 0.1, 2)])
        print(y_pred)
        tensor_input_x=torch.cat((tensor_input_x.to(device),y_pred.unsqueeze(0).unsqueeze(0).to(device)),dim=1)
    print(tensor_input_x)



    # 輸出output.csv
    # data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
    #         ["2018-01-01 01:00:00", "sell", 3, 5]]
    output(args.output, data)
    #torch.cuda.empty()





