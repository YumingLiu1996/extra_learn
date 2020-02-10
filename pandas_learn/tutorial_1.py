'''
此节用来学习pandas中loc以及iloc
loc以及iloc都是用来对行进行索引的
区别在于loc是根据行名来对行进行索引的
而iloc是通过行的序号来对行进行索引的
'''



import pandas as pd

data = [[1,2,3],[4,5,6],[7,8,9]]
index = ['d','e','f']
columns = ['a', 'b', 'c']
df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
## loc通过行标签索引行数据，返回一行时数据类型是series
print(df.loc['d'])
## loc可以索引多行数据，多行是返回的数据为dataframe
print(type(df.loc['d':]))
## loc扩展——索引某行某列,先确定行索引，再确定列索引
print(df.loc['d':,['b']])



## df iloc通过行号获取行数据
print(df.iloc[1])



