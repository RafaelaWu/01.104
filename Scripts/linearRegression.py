import numpy as np
import projectLib as lib
import copy
from matplotlib import pyplot as plt 

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    for i in range(trStats["n_ratings"]):
        A[i][training[i][0]] = 1  #movies
        A[i][training[i][1]+97] = 1
    return A


# we also get c
def getc(rBar, ratings):
    rate = copy.deepcopy(ratings)
    for i in range(len(ratings)):      
        rate[i] = rate[i] - rBar
    return rate

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    I = np.identity(trStats["n_movies"] + trStats["n_users"],dtype=None)
    AT = np.transpose(A)
    ATA = np.dot(AT,A)
    lI = 0.000000000001*I
    inverse = np.linalg.inv(ATA+lI) 
    ATc = np.matmul(AT,c)
    b = np.matmul(inverse,ATc)
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    I = np.identity(trStats["n_movies"] + trStats["n_users"],dtype=None)
    AT = np.transpose(A)
    ATA = np.matmul(AT,A)
    lI = l*I
    inverse = np.linalg.inv(ATA+lI)    
    ATc = np.matmul(AT,c)
    b = np.matmul(inverse,ATc)
    return b   



# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
#l =0
#b = param(A, c)

# Regularised version
l = min_regularization
b = param_reg(A, c, l)

print ("Linear regression, l = %f" % l)
print ("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print ("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))

##Qns 1
predictedRatings = np.array([predict(trStats["movies"], trStats["users"], rBar, b)])
################################################################################################
##Output for 300 by 97
def output(training,b):
    G = np.zeros((trStats["n_users"], trStats["n_movies"]))
    for user in range(300):
        for movie in range(97):
            rating = rBar + b[movie]+b[user+97]
            if rating > 5: rating = 5.0
            if rating < 1: rating = 1.0
            G[user][movie] = rating
    return G




min_rmse=10
min_regularization = 0
rmselist= []
#rrange = [0.001, 0.01, 0.1, 0.5, 1, 10]
rrange = np.linspace(2.52, 2.55, num=30)
for regularization in rrange:
    b = param_reg(A, c, regularization)
    #print "Linear regression, l = %f" % regularization
    rmse = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
    rmselist.append(rmse)
    #print rmse
    if rmse < min_rmse:
        min_rmse = rmse
        min_regularization = regularization

plt.plot(rrange, rmselist,'ro')

#plt.axis([6, 7, 1.069, 1.070])

plt.title("RMSE vs regularisation parameter")
print ("Minimum: %f,%f" % (min_regularization, min_rmse))
plt.show()



output = output(training,b)


np.savetxt("Linearregression+v1.csv", output)


