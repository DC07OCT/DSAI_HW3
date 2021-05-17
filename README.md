# DSAI_HW3
# 分析50個household train data 在各時段的consumption 和 generation (資料前處理)  
1.將terget0到target49拿出來看,發現資料最齊全的是有5831筆資料  
2.把target0到49分別generation和consumption,對齊整理兩個表格(缺失值補0)  
3.最後發現target0到49都 3/11 lose "2018/3/11  02:00:00 AM" 資料,所以全部補0   
將他們整理在一起得到各時段總和的consumption_0507.csv和generation_0507.csv(有5832筆) 做觀察  

# 用DataLoader包Train資料   
因為想用每七天的資料predict 下一小時,因此input_size為(7*24)=168筆[generation,consumption]而 output=[generation,consumption]x1   
* training dataset 共有50個targets(8個月資料/each)共236x24x50=283200筆training data   
* input data需要-mean/std   

# Model
使用LSTM with  
              input_dim = 2   
              hidden_dim = 128  
              num_layers = 2  
              output_dim = 64
 
# Training and Validation Method  
用MSELoss和ADAM optimizer Training   
batch_size=119->model_input_size=119x168x2且 model_output_size=119x1x2   
因為資料量太大,所以共training 30epoch(每個epoch有 283200/119=2400 batch)且每500個batch做一次validation   
* 每個epoch紀錄一次epoch_loss,訓練完成後print出來[epoch_num長]array  
 
![image](https://github.com/DC07OCT/DSAI_HW3/blob/main/dsai_hw3_training_process/5.png)

# 存模型參數  
'params.pkl'  

# Test Sample data  
input_data_size=1x168x2(7天)  
output_data_size=1x2(1小時)  
再用input_data[1:]+output_data做下一筆input,再下一筆input=input_data[2:]+output_data ..以此類推得到24hr predicted資料   

# 決策
* 台電=2.53(市場價格的 upper bound)  
* 數值是觀察平台info.csv資料下的  
* 整體的下標量不超過缺/多的量  
* 因為一次最多100筆下標所以一小時分配4個標籤   
ˋˋˋpython

        #if y_pred[0]==y_pred[1]:#do nothing  
        if y_pred[0]>y_pred[1]: #代表有餘電要賣(generation>consumption)  
            data.append([next_hour,"sell",2.48,round((y_pred[0]-y_pred[1]).tolist()*0.1,2)])  
            data.append([next_hour,"sell",2.35,round((y_pred[0]-y_pred[1]).tolist()*0.2,2)])  
            data.append([next_hour, "sell", 2.2,round((y_pred[0]-y_pred[1]).tolist()*0.7,2)])  
            data.append([next_hour, "sell", 2.0,round((y_pred[0]-y_pred[1]).tolist()*0.1,2)])  

        elif y_pred[0] < y_pred[1]: #代表電不足 要買(generation<consumption)   
            data.append([next_hour, "buy", 2.51, round((y_pred[1] - y_pred[0]).tolist() * 0.7, 2)])  
            data.append([next_hour, "buy", 2.4, round((y_pred[1] - y_pred[0]).tolist() * 0.2, 2)])  
            data.append([next_hour, "buy", 2.25, round((y_pred[1] - y_pred[0]).tolist() * 0.1, 2)])  
            data.append([next_hour, "buy", 2.1,round((y_pred[1] - y_pred[0]).tolist() * 0.1, 2)])
 ˋˋˋ
      
