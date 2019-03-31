import numpy as np
import rbm
import projectLib as lib

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5
alpha = 0.9

# SET PARAMETERS HERE!!!
# number of hidden units
F = 5
epochs = 30
gradientLearningRate = 0.0001
gradientLearningRate_v = 0.001
gradientLearningRate_h = 0.001
_lambda = 1
minibatch_size = 10

def main_rbm(training=training, validation=validation, trStats=trStats, vlStats=vlStats, 
             K=5, F=5, epochs=30, gradientLearningRate=0.0001,gradientLearningRate_v = 0.001,
             gradientLearningRate_h = 0.001, minibatch_size=10, alpha=0.9, 
             stopping=False, momentum=False, learning_rate_type='time', learning_rate_k=0.1, 
             learning_rate_drop=0.5, learning_rate_epochs_drop=10.0, _lambda = 0.3):
    
    # Print current hyperparams
#     frame = inspect.currentframe()
#     args, _, _, values = inspect.getargvalues(frame)
#     print ('Training and Predicting with the following hyperparameters:')
#     for i in args[4:]:
#         print ("    %s = %s" % (i, values[i]))
        
    # Initialise all our arrays
    num_movies=trStats["n_movies"]
    num_users=trStats["n_users"]
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)
    posprods = np.zeros(W.shape)
    negprods = np.zeros(W.shape)
    grad_w = np.zeros(W.shape)
    m_w=np.zeros((W.shape[0],F,5))
    train_loss = []
    validation_loss = []
    vis_bias=np.zeros((num_movies,5))
    m_v=np.zeros((num_movies,5))
    hid_bias=np.zeros((F,))
    m_h=np.zeros((F,))
    best_train_loss = 100
    best_validation_loss = 100
    
    for epoch in range(1, epochs+1):
    #     mini_batch_grads = []
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)
#         for i in range(0, visitingOrder.shape[0], minibatch_size):
#                 # Get pair of (X, y) of the current minibatch/chunk
#             visitingOrderMini = visitingOrder[i:i + minibatch_size]
#                 y_train_mini = y_train[i:i + minibatch_size]
#         for i in range(0, visitingOrder.shape[0], minibatch_size):
        batches = batch_get(visitingOrder, minibatch_size)
        for batch in batches:
            prev_grad = grad_w
            grad_w = np.zeros(W.shape)
            for user in batch:
                # get the ratings of that user
                ratingsForUser = lib.getRatingsForUser(user, training)

                # build the visible input
                v = rbm.getV(ratingsForUser)

                # get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :]

                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser) #, hid_bias)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] += probProduct(v, posHiddenProb)

                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)#, vis_bias[ratingsForUser[:,0]])
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser) #, hid_bias)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)

                poshidact = sum(posHiddenProb)
                posvisact = sum(v)
                neghidact = sum(negHiddenProb)
                negvisact = sum(negData)
                # we average over the number of users in the batch (if we use mini-batch)
    #             grad = (gradientLearningRate/epoch)*(posprods-negprods)
                '''
                Regularization - 
                '''
                grad_w += adaptiveLearn(learning_rate_type=learning_rate_type, k=learning_rate_k, 
                                        drop=learning_rate_drop, epochs_drop=learning_rate_epochs_drop, 
                                        epoch=epoch)*((posprods-negprods)/trStats["n_users"]-_lambda*W)
        #         mini_batch_grads.append(grad)

            #     m = alpha*m+grad
                '''
                Ask about the implementation of biases (should we create matrix of biases for hidden and visible layers?)
                '''
            m_w = alpha*m_w + grad_w
#             m_v = alpha*m_v+(gradientLearningRate_v) * (posvisact - negvisact)
#             m_h = alpha*m_h+(gradientLearningRate_h) * (poshidact - neghidact)

            if momentum == False:
                W += grad_w
            else:
                W += m_w

            vis_bias += m_v
            hid_bias += m_h

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training) #, vis_bias, hid_bias, predictType='exp')
    #     print (tr_r_hat)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
    #     print (trRMSE)
        if trRMSE < best_train_loss:
            best_train_loss = trRMSE
            best_training_weights = W
            best_train_predictions = tr_r_hat

        # We predict over the validation set
        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training) #, vis_bias, hid_bias, predictType='exp')
    #     vl_r_hat
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
        if vlRMSE < best_validation_loss:
            best_validation_loss = vlRMSE
            best_validation_weights = W
            best_validation_predictions = vl_r_hat
#             best_momentum = momentum
#             best_reg = regularization
#             best_epoch = epoch
#             best_alpha = alpha
#             best_B = B
#             best_F = F
#             min_rmse = vlRMSE
#                     print('Best RMSE:', min_rmse)

        train_loss.append(trRMSE)
        validation_loss.append(vlRMSE)

        print ("### EPOCH %d ###" % epoch)
        print ("Training loss = %f" % trRMSE)
        print ("Validation loss = %f" % vlRMSE)

    ### END ###
    # This part you can write on your own
    # you could plot the evolution of the training and validation RMSEs for example
    # predictedRatings = np.array([predictForUser(user, W, training) for user in trStats["u_users"]])
    # np.savetxt("predictedRatings.txt", predictedRatings)
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(121)
    # ax1.plot(train_loss)
    # ax2 = fig1.add_subplot(122)
    # ax2.plot(validation_loss)
    
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.show()
    if stopping==True:
        print('Best training loss = %f' % best_train_loss)
        print('Best validation loss = %f' % best_validation_loss)
    else:
        print('Final training loss = %f' % trRMSE)
        print('Final validation loss = %f' % vlRMSE)
    return [best_validation_loss, best_validation_predictions]

#Hyperparameter tuning:

# mrange = np.linspace(0.7,0.95,5)
mrange = [0.9]
# rrange = np.linspace(0.1,0.9,5)
rrange = [0.001, 0.01, 0.1, 0.5, 1]
arange = [0.0001, 0.001, 0.01, 0.1]
brange = [10]
frange = [8]

best_momentum = 0
best_reg = 0
best_alpha = 0
best_batch = 0
best_F = 0

min_rmse=10
for momentum in mrange:
    for regularization in rrange:
        for learning_rate in arange:
            for batch in brange:
                for F in frange:
                    prediction = main_rbm(training=training, validation=validation, trStats=trStats, vlStats=vlStats, 
                                 K=5, F=int(F), epochs=30, gradientLearningRate=learning_rate, gradientLearningRate_v = 0.001,
                                 gradientLearningRate_h = 0.001, minibatch_size=int(batch), alpha=momentum, 
                                 stopping=True, momentum=True, learning_rate_type='time', learning_rate_k=0.5, 
                                 learning_rate_drop=0.5, learning_rate_epochs_drop=10.0, _lambda = regularization)
                    if prediction[0] < min_rmse:
                        best_momentum = momentum
                        best_reg = regularization
#                             best_epoch = epoch
                        best_alpha = learning_rate
                        best_batch = batch
                        best_F = F
                        best_predict = prediction[1]
                        min_rmse = prediction[0]

## Final parameters:
alpha = 0.9
_lambda = 0.1 #tuned
gradientLearningRate = 0.001 #tuned
minibatch_size = 10
F = 8 

# TODO: Tune other parameters and add biases