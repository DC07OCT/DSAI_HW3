# DSAI_HW3
# 分析50個household train data 在各時段的consumption 和 generation (資料前處理)
1.將terget0~target49拿出來看,發現資料最齊全的是有5831筆資料
2.把target0~49分別generation和consumption,對齊整理兩個表格(缺失值補0)
3.最後發現target0~49都 3/11 lose "2018/3/11  02:00:00 AM" 資料,所以全部補0
將他們整理在一起得到各時段總和的consumption_0507.csv和generation_0507.csv(有5832筆) 做觀察
# 用DataLoader包Train資料
因為想用每七天的資料predict 下一小時,因此input_size為(7*24)=168筆[generation,consumption]而 output=[generation,consumption]*1
* training dataset 共有50個targets(8個月資料/each)共236*24*50=283200筆training data
* input data需要-mean/std

# Model
使用LSTM with
# Training and Validation Method
batch_size=119->model_input_size=119*168*2且 model_output_size=119*1*2
因為資料量太大,所以共training 30epoch(每個epoch有 283200/119=2400 batch)且每500個batch做一次validation
* 每個epoch紀錄一次epoch_loss
result



