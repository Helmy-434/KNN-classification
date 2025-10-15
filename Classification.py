from collections import Counter

def classification (train_x,train_y,tests_x,tests_y, k):
    predicted_output=[]

    for test in tests_x:
        distances = []

        for i in range(len(train_x)):
            distances.append( [ distance(test,train_x[i]), train_y[i] ] ) # get all the distances between each test case and all the training data

        distances.sort() # sort ascendingly to get the closest neighbours
        labels = []

        for i in range(k):
            labels.append(distances[i][1]) #get the labels of the closest neighbours only

        predicted_output.append( Counter(labels).most_common(1)[0][0] ) #add the most common label from the closest neighbours to the predicted outputs

    i=0
    correct=0
    for output in predicted_output :
        if output == tests_y[i]:
            correct +=1
        i+=1
    accuracy=correct/len(tests_y)
    print("At k = " + str(k) + " accuracy = " +str(accuracy*100)+"%")

def distance (x,y):
    n=len(x)
    d=0
    for i in range(n):
        d += (x[i]-y[i]) ** 2
    return d ** 0.5