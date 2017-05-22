# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:10:29 2017

@author: Artem
"""
import csv
import pickle
import random

def separate(database):
    y = []
    x = []
    y_train = []
    x_train = []
    part = len(database) //100 *10
    cnt=0
    for point in database:
        if len(point[0])==1:
            if cnt>part:
                y.append(point[0])
                x.append(point[1:])
            else:
                y_train.append(point[0])
                x_train.append(point[1:])
        else:
            pass
        cnt+=1 
            
            
    #print(y)
    return x, y, x_train, y_train

def separate_roman(database):
    y = []
    x = []
    y_train = []
    x_train = []
    roman = 0
    for i in database:
        if i[0] == '1':
            roman+=1
    part = len(database) //100 *10
    romancnt = 0
    roman-=500
    cnt=0
    
    for point in database:
        if len(point[0])==1:
            if point[0] == '1':
                if romancnt<=roman:
                
                    if cnt>part:
                        y.append(point[0])
                        x.append(point[1:])
                    else:
                        y_train.append(point[0])
                        x_train.append(point[1:])
                    romancnt+=1
            else:
                if cnt>part:
                    y.append(point[0])
                    x.append(point[1:])
                else:
                    y_train.append(point[0])
                    x_train.append(point[1:])
                
        elif point is None:
            break
        else:
            pass
        print(cnt)
        cnt+=1
        
    return x, y, x_train, y_train
    
def main():
    '''
    database=[]
    with open('data.csv', newline='') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader: 
            new =[]
            newitem = ''
            for entry in row:
                if entry != ',':
                    newitem += entry
                else:
                    new.append(newitem)
                    newitem=''
            if len(newitem) != 0:
                database.append(new)
    random.shuffle(database, random.random)
    data = open('database.txt', 'wb')
    print(database)
    pickle.dump(database, data)
    '''
    
    
    datao = open('database.txt', 'rb')
    
    f = pickle.load(datao)
    '''
    for line in range(len(f)):
        print(line)
        print('_____')
        print(f[line][0])
    '''
    print(len(f))
    x, y, x_train, y_train = separate(f)
    
    #print(x, y)
    model = svm.SVC(C = 5, kernel="rbf", gamma = 5/3500)
    model.fit(x, y)
    prediction = model.predict(x_train)

    print("Score:" , model.score(x_train,y_train))
    cvscores = cross_val_score(model, x, y, cv = 5)
    
    print("Accuracy: %0.2f (+/- %0.2f)" % (cvscores.mean(), cvscores.std() * 2))
    print(model.get_params())
main()

