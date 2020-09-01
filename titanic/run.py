import pandas as pd
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(test.iloc[1])

train_X = list(train['Pclass'])
train_Y = list(train['Survived'])

test_X = list(test['Pclass'])
# test_Y = list(test['Survived'])


train_x = [[x] for x in train_X]
train_y = [[y] for y in train_Y]

test_x = [[x] for x in test_X]
# test_y = [[y] for y in test_Y]


model.fit(train_x, train_y)

prediction = model.predict(test_x)

result = {'PassengerId':list(test['PassengerId']),'Survived':list(prediction)}

pd.DataFrame(result).to_csv('result.csv',index=None)