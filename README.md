# DSAI_HW3
# 分析50個household train data 在各時段的consumption 和 generation (資料前處理)
1.將terget0~target49拿出來看,發現資料最齊全的是有5831筆資料
2.把target0~49分別generation和consumption,對齊整理兩個表格(缺失值補0)
3.最後發現target0~49都 3/11 lose "2018/3/11  02:00:00 AM" 資料,所以全部補0


將他們整理在一起得到各時段總和的consumption和generation  
3/11 lose   2018/3/11  02:00:00 AM
