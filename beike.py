import pandas as pd

data = pd.read_excel('beike_data.xls')

xueduan = '小学'
xueke = '数学'
jiaocai = '人教版2013版'
nianji = '三年级'
kechengming = '5 倍的认识'

filtered_data = data.loc[data['学段'] == xueduan]
filtered_data = filtered_data.loc[filtered_data['学科'] == xueke]
filtered_data = filtered_data.loc[filtered_data['教材'] == jiaocai]
filtered_data = filtered_data.loc[filtered_data['年级'] == nianji]
filtered_data = filtered_data.loc[filtered_data['章节'] == kechengming]

print(filtered_data['id'])