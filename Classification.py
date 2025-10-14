from collections import Counter

def classification (tests_x,tests_y,train_x,train_y, k):
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


def distance (x,y):
    #    n=x.size



