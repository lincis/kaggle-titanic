import pandas
import seaborn as sns
import os
#localpath = os.path.dirname(os.path.realpath(__file__))

traindata = pandas.read_csv('Documents/kaggle-titanic/data/train.csv')
#print(traindata.head(10))

s_by_class = traindata[['Survived','Pclass']].groupby(['Pclass']).agg(['mean'])
s_by_class.plot()

s_by_sex = traindata[['Survived','Sex']].groupby(['Sex']).agg(['mean'])
s_by_sex.plot()

traindata['Age_cut'] = pandas.cut(traindata['Age'],[0,3,14,25,65,100])

s_by_age = traindata[['Survived','Age_cut']].groupby(['Age_cut']).agg(['mean'])
s_by_age.plot()

s_by_sibsp = traindata[['Survived','SibSp']].groupby(['SibSp']).agg(['mean'])
s_by_sibsp.plot()

s_by_parch = traindata[['Survived','Parch']].groupby(['Parch']).agg(['mean'])
s_by_parch.plot()

traindata['Fare_cut'] = pandas.cut(traindata['Fare'],[0,5,50,100,1000])
s_by_fare = traindata[['Survived','Fare_cut']].groupby(['Fare_cut']).agg(['mean'])
s_by_fare.plot()

s_by_emb = traindata[['Survived','Embarked']].groupby(['Embarked']).agg(['mean'])
s_by_emb.plot()