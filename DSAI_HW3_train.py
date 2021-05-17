# Training model
# You should not modify this part.
import pandas as pd
import argparse
import torch
import torch.nn as nn
import numpy as np

def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

c_mean=1.443836
c_std=1.630945
g_mean=0.780310597
g_std=1.413282181

from torch.utils.data import Dataset, DataLoader
class TrainDataset(Dataset):
    def __init__(self,gen_path,con_path):
        self.gen_path=gen_path #'/generation_0507.csv'
        self.con_path=con_path #'/consumption_0507.csv'
        self.n_samples=50*236*24
        # read csv
        generation = pd.read_csv(self.gen_path)
        consumption = pd.read_csv(self.con_path)

        time_series = generation['Time']
        i = 0  # initial 資料起點
        x_input = []  # 最後input資料包(236*50個 )
        y_input = []  # 最後output資料包
        while i < 5641:  # 下面還要有8天資料的最大index=5840
            flag = 1
            now = time_series[i][:9]  # 取日期部分
            #print('*' + now + '*')
            j = i
            x_seven = []  # 168個
            y_seven = []

            while j < 5832:
                if flag <= 7:
                    if not (now in time_series[j]):  # 取not
                        flag = flag + 1
                        if flag == 2:
                            i = j  # 更改下一個起頭點
                        now = time_series[j][:9]
                    else:  # now in time_series[j]
                        # 放入input data
                        tem1 = []  # 放TARGET0~49
                        for k in range(50):
                            tem2 = []  # 放[generation,consumption]
                            tem2.append((generation['target' + str(k)][j]-g_mean)/g_std)# 一定要做
                            tem2.append((consumption['target' + str(k)][j]-c_mean)/c_std)
                            tem1.append(tem2)
                        x_seven.append(tem1)  # 放7天資料
                        j = j + 1

                else:  # flag==8:
                    if (now in time_series[j]):
                        # 放入GT(放入target0~49)
                        tem1 = []
                        for k in range(50):
                            tem2 = []
                            tem2.append((generation['target' + str(k)][j]-g_mean)/g_std)
                            tem2.append((consumption['target' + str(k)][j]-c_mean)/c_std)
                            tem1.append(tem2)
                        y_seven.append(tem1)
                        j = j + 1
                    else:  # 第8天取完後
                        y_input.append(y_seven)  # 最後input資料包的ground truth
                        x_input.append(x_seven)  # 最後input資料包
                        # flag=1
                        break
        y_input.append(y_seven)  # 最後一組(236th)進不到else(第8天取完後)
        x_input.append(x_seven)

        # print('len(x_input)',len(x_input))#236?

        tem_x_input = []
        for i in range(len(x_input)):  # row data轉column
            I = zip(*x_input[i])
            T = [list(k) for k in list(I)]
            tem_x_input.append(T)
        I = zip(*tem_x_input)
        final_x_input = [list(k) for k in list(I)]  # [[[[gen,con]*168]*236]*50]??

        tem_y_input = []
        for i in range(len(y_input)):  # row data轉column
            I = zip(*y_input[i])
            T = [list(k) for k in list(I)]
            tem_y_input.append(T)
        I = zip(*tem_y_input)
        final_y_input = [list(k) for k in list(I)]  # [[[[gen,con]*168]*236]*50]

        self.re_x_input = []  # 把[target0][target1][target2]...的[]打開 且 改成包成一小時為單位training分24小時
        for i in range(len(final_x_input)):#target0~49
            for j in range(len(final_x_input[i])):  # 7 day(7*24=168)
                for k in range(24):  # each hour
                    temp = final_x_input[i][j][k:]+final_y_input[i][j][:k]
                    self.re_x_input.append(temp)#temp= size =[[generation,consumption]*168]



        self.re_y_input = []
        for i in range(len(final_y_input)):
            for j in range(len(final_y_input[i])):
                for k in range(24):
                    temp=[]
                    temp.append(final_y_input[i][j][k][0]*g_std+g_mean)#*g_std+g_mean
                    temp.append(final_y_input[i][j][k][1]*c_std+c_mean)#*c_std+c_mean
                    self.re_y_input.append(temp) #final_y_input[i][j][k] size 為 [[generation,consumption]*1]





    def __getitem__(self, index):
        """ Changing list to tensor. """
        list_of_tensors=[torch.tensor(np.array(i)) for i in self.re_x_input[index]] #self.re_x_input[index]為list [[gen,con]*168]
        self.re_x_input[index] =torch.stack(list_of_tensors)
        list_of_tensors = [torch.tensor(np.array(i)) for i in self.re_y_input[index]]#self.re_y_input[index]為list [[gen,con]*24]
        self.re_y_input[index] = torch.stack(list_of_tensors)
        #168*2大小

        return self.re_x_input[index],self.re_y_input[index]

    def __len__(self):
        return self.n_samples #50*236*24



class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(LSTM, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.fc1=nn.Linear(output_dim,2)
                self.relu=nn.ReLU(inplace=True)

        def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

                out1 = self.relu(self.fc(hn[1]))#for generation #out size=(batch size,len_seq,hidden_dim)
                out2 = self.fc1(out1)#for consumption   # out size=(batch size,len_seq,hidden_dim)
                return out2






def output(path, data):


    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()
    # add your model here

    #data loader
    dataset = TrainDataset('Z:/generation_0507.csv',
                           'Z:/consumption_0507.csv')
    b_size=118
    loader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=False)  # 236為一個target的資料,原236

    # model parameters
    input_dim = 2  # 放入 [generation,consumption] 做training #input_dim是指輸入維度
    hidden_dim = 128  # 代表一層hidden layer有128個LSTM neuron
    num_layers = 2  # 2層hidden layer
    output_dim = 64
    num_epochs = 30

    device = torch.device('cuda'if torch.cuda.is_available() else "cpu")

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    hist = np.zeros(num_epochs)  # 用來記錄歷史值

    #Training Process
    model=model.float()
    for t in range(num_epochs):
        epoch_loss=0
        count=0
        for i, (x, y) in enumerate(loader):  # 要再切validation( start=1 是指數字從1開始但還是50batches,沒必要)
            if (i%500)!=0:
                model.train()
                x = x.to(device)
                y = y.to(device)


                y_pred = model(x.float())
                loss = torch.div(criterion(y_pred.float(), y.float()),float(b_size)).requires_grad_()#torch.stack做tensor的zip
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            else:#Validation
                count=count+1
                model.eval()
                x = x.to(device)
                y = y.to(device)
                y_eval = model(x.float())
                print(y_eval)
                loss_eval = criterion(y_eval.float(), y.float())
                print('epoch',t,'batch',i,'eval',loss_eval.item()/b_size)

        epoch_loss = epoch_loss / (len(loader)-count) #每一個 predict的loss(原有)

        print(epoch_loss)
        hist[t] = epoch_loss
        if epoch_loss < 0.01:  # 提早跳出epoch
            break
    print(hist)

   #存training model 參數
    torch.save(model.state_dict(),'params.pkl')
