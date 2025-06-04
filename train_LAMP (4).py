

import random
import pickle
#import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import to_categorical
from art.utils import load_mnist, load_cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import resnet50
from resnet_wrn import resnet
from sam import SAM

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, CarliniLInfMethod, DeepFool, SaliencyMapMethod, BasicIterativeMethod
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from sklearn.metrics import accuracy_score

# Name of model being trained
#name = './00_models/02_TEST_padnet_CIFAR100=True,TGR=ON,alpha=10,padding=ON'
name = 'Adv_Mix_Addaptive_PGD

# Load CIFAR-10
(X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
X_train = X_train.reshape(X_train.shape[0],3072)

X_train = X_train.reshape(X_train.shape[0],3072)
#print("y_train examples", y_train[:20])
# Load Padding CLass
#(X_train, y_train) = pickle.load( open( "CIFAR10_with_padding_class.pkl", "rb" ) )

"""
# Load PGD Adversarial Examples
###########################################
(X_train_adv, y_train_adv) = pickle.load( open( "PGD_examples.pkl", "rb" ) )
X_train_adv = X_train_adv.reshape(X_train_adv.shape[0],3072)
y_train_adv = y_train_adv.astype(int)

# load Adaptive Examples
#########################################
path = '../adaptive_attack/CIFAR10/'
(X_train_adapt, y_train_adapt) = pickle.load( open( path + "adaptive_examples.pkl", "rb" ) )
X_train_adapt = X_train_adapt.reshape(X_train_adapt.shape[0],3072)
y_train_adapt = y_train_adapt.astype(int)


# Set PGD AEs as the padding class
for i in range(0,y_train_adv.shape[0]):
    y_train_adv[i] = 10
    
# Set Adaptive Examples as the padding class
for i in range(0,y_train_adapt.shape[0]):
    y_train_adapt[i] = 10
    
"""
        
#X_train = np.concatenate([X_train, X_train_adv])
#y_train = np.concatenate([y_train, y_train_adv])

#shuffler = np.random.permutation(X_train.shape[0])
#X_train = X_train[shuffler]
#y_train = y_train[shuffler]
###########################################

#print(X_train.shape)
#print(y_train.shape)
X_train = X_train.reshape(X_train.shape[0],32,32,3)
X_test = X_test.reshape(X_test.shape[0],32,32,3)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)


  

##################################################
# Wide ResNet CNN 
##################################################

My_wd=5e-4/2
resnet_width = 6#10 #10
resnet_depth = 12#34 #20
UseBinary=False
input_shape = [32,32,3]
num_classes = 11

model = resnet(UseBinary,input_shape=input_shape, depth=resnet_depth, num_classes=num_classes,wd=My_wd,width=resnet_width)
#model = keras.applications.resnet50.ResNet50(
#    include_top=False, input_tensor=None, 
#    input_shape=(32, 32, 3), pooling=None, classes=101)
#######################################################################
# End Wide ResNet CNN
########################################################################
    # Ed Altered to replace with RESNET 50 preprocessing
def preprocess(image):
    #image = keras.applications.resnet50.preprocess_input(image, data_format='channels_last')
    image = tf.cast(image, tf.float32)
#    #image = tf.image.resize(image, (8, 8))
#    #image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#    #image = image[None, ...]
    return image    


def early_stopping(model, X_train, y_train, X_test, y_test, name, adv_grad_norm, grad_norm):
    
  
    
    
    model.layers[-1].activation=None
    
    num_test_samples = 15
    X_test_small = X_test[100:100+num_test_samples]
    y_test_small = y_test[100:100+num_test_samples]
    
    loss = keras.losses.CategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(model=model, clip_values=(0,1), nb_classes=11, input_shape=(32, 32, 3), loss_object=loss)
    #attack = CarliniLInfMethod(classifier=classifier, confidence=0, max_iter=10, targeted=False)
    attack = CarliniL2Method(classifier=classifier, confidence=0, initial_const=1, max_iter=10, targeted=False)
    x_test_adv = attack.generate(x=X_test_small)
    predictions = classifier.predict(x_test_adv)
    y_pred = np.argmax(predictions, axis=1)
    
    success = 0.0
    for i in range(0,y_test_small.shape[0]):
        #print(y_pred[i], y_test[i])
        if y_pred[i] != y_test_small[i] and y_pred[i] != 10:
            success += 1.0
    sr = success/y_test_small.shape[0]
    print("attack_success: "+ str(sr))
    f = open(name+".txt", "a")
    f.write("Attack success: " + str(sr) +"\n")
    f.close()
    model.layers[-1].activation=tf.keras.activations.softmax
    
    y_pred = model(X_train)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_train, y_pred)
    print("Train Accuracy:", acc)
    f = open(name+".txt", "a")
    f.write("Train Accuracy: " + str(acc) +"\n")
    f.close()
    
    y_pred = model(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    f = open(name+".txt", "a")
    f.write("Test Accuracy: " + str(acc) +"\n")
    f.close()
    
    print("adv_grad_norm:", adv_grad_norm)
    f = open(name+".txt", "a")
    f.write("adv_grad_norm: " + str(adv_grad_norm) +"\n")
    f.close()
    
    print("grad_norm:", grad_norm)
    f = open(name+".txt", "a")
    f.write("grad_norm: " + str(grad_norm) +"\n")
    f.close()
    
    return sr, acc



softmax = tf.keras.layers.Softmax()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
cross_entropy = tf.keras.losses.CategoricalCrossentropy()

def PGD(X_train, y_train):

    X_train = X_train.reshape(X_train.shape[0],32,32,3)
    # add noise for starting point of adversarial examples
    noise = np.random.normal(0, 1, X_train.shape)
    x_adv = X_train + .001 * noise
            
    adv_steps = 10 #10 
    adv_steps == random.randint(1,10)
    adv_step_size = .007 #.007 
    #epsilon = .031 #.3 #.031
    epsilon = random.uniform(.01,.9) #(0.01,0.031)
    
    label_flag = random.randint(0,1)
    
    for a in range(0, adv_steps):
                
        with tf.GradientTape() as tape:
            x_adv = preprocess(x_adv)
            tape.watch(x_adv)
            prediction = model(x_adv) # This model should be a fully trained model
            #natural = model(X_batch)
            #loss = cross_entropy(softmax(natural), softmax(prediction))
            #loss = kl_div(softmax(natural), tf.nn.log_softmax(prediction))
            #loss = cross_entropy(y_batch, prediction)
            pad = [10] * len(x_adv)
            pad = np.asarray(pad)
            if label_flag == 0:
                loss = loss_object(y_train, prediction) #works best
            else:
                loss = loss_object(pad, prediction)
        
                

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, x_adv)
           
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
           
                
        
        if label_flag == 0:
            x_adv = x_adv + adv_step_size * signed_grad
        else:
            x_adv = x_adv + adv_step_size * signed_grad
        
        x_adv = x_adv.numpy()
            
        x_adv = np.clip(x_adv, X_train - epsilon, X_train + epsilon)
                
        x_adv = np.clip(x_adv, 0, 1)
    
    x_adv = x_adv.reshape(x_adv.shape[0],3072)
    return x_adv

def gen_lamp_images(X_train, y_train):
    
    a = X_train[0]
    c10 = np.copy(a) # append class 10 to this ndarray
    l = y_train[0]
    l10=np.copy(l) # append class 10 labels to this ndarray
    #print("whole list: ", l10,"\n last one: ",l)
    
    x_adv = PGD(X_train, y_train)
    
    size = X_train.shape[0] 
    for i in range(size -1):
        
        ############## Init source and target ###########
        s = np.copy(X_train[i])
        t = x_adv[i]
        s_t = np.vstack ((s, t))
        #s_t = np.vstack ((s, X_train[i+1] ))
        
        ############### Find mean ################
        m = np.mean(s_t, axis=0)
        
        #################### Add uniform padding examples ############
        #alpha = .2
        alpha = random.uniform(0.05,0.95)
        wa1 = np.average(s_t, axis=0, weights=[alpha, 1-alpha])
        
        #var = random.uniform(0.01,0.1) #.05,.1
        #rand = np.random.normal(0, var, 3072)
        #rand = rand.reshape(3072,)
        #wa1 = np.add(rand,wa1)
        #wa1 = np.clip(wa1,0,1)
        c10 = np.vstack ((c10,wa1))
        l10 = np.vstack ((l10,10)) #edit here from 10 to 100 for CIFAR 100
        #l10 = np.vstack ((l10,y_train[i]))
        
        #################### Add extensive gaussian noise to median to create "noise class" #############
        #var = random.uniform(0.01,0.1) #.05,.1
        var = .01
        rand = np.random.normal(0, var, 3072)
        rand = rand.reshape(3072,)
        rand_pert = np.add(rand,m)
        rand_pert = np.clip(rand_pert,0,1)
        c10 = np.vstack((c10,rand_pert))
        #c10 = np.vstack((c10,m))
        l10 = np.vstack((l10,10)) #edit here from 10 to 100 for CIFAR 100
        #l10 = np.vstack ((l10,y_train[i]))
        
    # add one more benign sample to mach size
    c10 = np.vstack ((c10, X_train[1]))
    l10 = np.vstack ((l10, y_train[1]))
    
    l10 = l10.reshape(l10.shape[0],)
    c10 = c10.reshape(c10.shape[0],32,32,3)
    #print("whole list: ", l10)
    return c10, l10
    


def gen_padding_images(X_train, y_train):
    
    a = X_train[0]
    c10 = np.copy(a) # append class 10 to this ndarray
    l = y_train[0]
    l10=np.copy(l) # append class 10 labels to this ndarray
    #print("whole list: ", l10,"\n last one: ",l)
    
    
    size = X_train.shape[0] 
    for i in range(size -1):
        
        ############## Init source and target ###########
        s = np.copy(X_train[i])
        s_t = np.vstack ((s, X_train[i+1] ))
        
        ############### Find mean ################
        m = np.mean(s_t, axis=0)
        
        #################### Add uniform padding examples ############
        alpha = .2
        #alpha = random.uniform(0.1,0.9)
        wa1 = np.average(s_t, axis=0, weights=[alpha, 1-alpha])
        
        #var = random.uniform(0.01,0.1) #.05,.1
        var = .01
        rand = np.random.normal(0, var, 3072)
        rand = rand.reshape(3072,)
        wa1 = np.add(rand,wa1)
        wa1 = np.clip(wa1,0,1)
        c10 = np.vstack ((c10,wa1))
        l10 = np.vstack ((l10,10)) #edit here from 10 to 100 for CIFAR 100
       
        #################### Add extensive gaussian noise to median to create "noise class" #############
        #var = random.uniform(0.01,0.1) #.05,.1
        var = .01
        rand = np.random.normal(0, var, 3072)
        rand = rand.reshape(3072,)
        rand_pert = np.add(rand,m)
        rand_pert = np.clip(rand_pert,0,1)
        c10 = np.vstack((c10,rand_pert))
        l10 = np.vstack((l10,10)) #edit here from 10 to 100 for CIFAR 100
    
    # add one more benign sample to mach size
    c10 = np.vstack ((c10, X_train[1]))
    l10 = np.vstack ((l10, y_train[1]))
    
    l10 = l10.reshape(l10.shape[0],)
    c10 = c10.reshape(c10.shape[0],32,32,3)
    #print("whole list: ", l10)
    return c10, l10

def random_batch(X, y, batch_size):
    idx = np.random.randint(X.shape[0], size=batch_size)
    return X[idx], y[idx]

def print_status_bar(iteration, total, loss, acc, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) +"Test Accuracy: "+ acc +" "+ metrics,
          end=end)

datagen = ImageDataGenerator(
    
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    #brightness_range=[0.5,1.5],
    #zoom_range=[0.5,1.0],
    )
    
def train():
    
    # hyperparams
    n_epochs = 1000
    batch_size = 128
    lr = .001
    n_steps = X_train.shape[0] // batch_size
    optimizer = Adam(learning_rate=lr, beta_1=0.9)
    optimizer = SAM(optimizer)
    #optimizer = Nadam(learning_rate=0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.Accuracy()]
    best_sr = 1.0
    best_acc = 0
    count = 0
    alpha = 10
    beta = 1
    
    # training loop
    for epoch in range(1, n_epochs + 1):
        print("\nEpoch {}/{}".format(epoch, n_epochs))
        #for step in range(1, n_steps + 1):
        step = 1
        for X_batch, y_batch in datagen.flow(X_train, y_train, 32): # put back 4 if not undefended
            
            # add boundary padding
            X_pad, y_pad = random_batch(X_train, y_train, 32)
            X_pad = X_pad.reshape(X_pad.shape[0],3072)
            X_pad, y_pad = gen_lamp_images(X_pad, y_pad)
            X_batch = np.concatenate([X_batch, X_pad])
            y_batch = np.concatenate([y_batch, y_pad])
            #shuffler = np.random.permutation(X_batch.shape[0])
            #X_batch = X_batch[shuffler]
            #y_batch = y_batch[shuffler]
            
            # add original samples
            X_orig, y_orig = random_batch(X_train, y_train, 32)
            X_batch = np.concatenate([X_batch, X_orig])
            y_batch = np.concatenate([y_batch, y_orig])
            shuffler = np.random.permutation(X_batch.shape[0])
            X_batch = X_batch[shuffler]
            y_batch = y_batch[shuffler]
            
            # add PGD Adversarial Examples
            #X_adv, y_adv = random_batch(X_train_adv, y_train_adv, int(batch_size / 8)) 
            #X_adv = X_adv.reshape(X_adv.shape[0], 32, 32, 3)
            #X_batch = np.concatenate([X_batch, X_adv])
            #y_batch = np.concatenate([y_batch, y_adv])
            #shuffler = np.random.permutation(X_batch.shape[0])
            #X_batch = X_batch[shuffler]
            #y_batch = y_batch[shuffler]
            
            # add Adaptive Adversarial Examples
            #X_adapt, y_adapt = random_batch(X_train_adapt, y_train_adapt, int(batch_size / 8)) 
            #X_adapt = X_adapt.reshape(X_adapt.shape[0], 32, 32, 3)
            #X_batch = np.concatenate([X_batch, X_adapt])
            #y_batch = np.concatenate([y_batch, y_adapt])
            #shuffler = np.random.permutation(X_batch.shape[0])
            #X_batch = X_batch[shuffler]
            #y_batch = y_batch[shuffler]
            
            # solve for adversarial gradient
            with tf.GradientTape() as tape:
                X_batch = preprocess(X_batch)
                tape.watch(X_batch)
                prediction = model(X_batch) 
                loss = loss_object(y_batch,prediction)
            grad_adv = tape.gradient(loss, X_batch)
            
            # solve for adversarial gradient
            with tf.GradientTape() as tape:
                X_batch = preprocess(X_batch)
                tape.watch(X_batch)
                prediction = model(X_batch) 
                pad = [10] * batch_size
                pad = np.asarray(pad)
                loss = loss_object(pad,prediction)
            grad = tape.gradient(loss, X_batch)
            
            with tf.GradientTape() as tape:
                predictions = model(X_batch, training=True)
                loss = loss_object(y_batch, predictions) # TGR OFF
                #loss = loss_object(y_batch, predictions) + (alpha * tf.math.reduce_mean(tf.math.square(grad))) # TGR ON
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.first_step(gradients, model.trainable_variables)
            
            
            # solve for adversarial gradient
            with tf.GradientTape() as tape:
                X_batch = preprocess(X_batch)
                tape.watch(X_batch)
                prediction = model(X_batch) 
                pad = [10] * batch_size
                pad = np.asarray(pad)
                loss = loss_object(pad,prediction)
            grad = tape.gradient(loss, X_batch)

            with tf.GradientTape() as tape:
                predictions = model(X_batch, training=True)
                loss = loss_object(y_batch, predictions) # TGR OFF
                #loss = loss_object(y_batch, predictions) + (alpha * tf.math.reduce_mean(tf.math.square(grad))) # TGR ON
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.second_step(gradients, model.trainable_variables)
            #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            mean_loss(loss)
            #print(X_test.shape)
            y_pred = model(X_test)
            #print(y_pred.shape)
            y_pred = np.argmax(y_pred, axis=1)
            #print(y_pred.shape)
            ##########################################################
            #print("y_pred = ", y_pred, "\ny_test = ", y_test)
            acc = accuracy_score(y_test, y_pred)
            print_status_bar(step * batch_size, y_train.shape[0], mean_loss, str(acc))
            
            
            if step % 150 == 0:
                adv_grad_norm = tf.math.reduce_mean(tf.math.square(grad_adv)).numpy()
                grad_norm = tf.linalg.global_norm(gradients).numpy()
                sr, acc = early_stopping(model, X_train[:1000], y_train[:1000], X_test, y_test, 'log/'+name, adv_grad_norm, grad_norm)
                name_new = "models/" + name + "_attack_success=" + str(sr) + "_test_acc=" + str(acc)
                model.save(name_new)
                if acc > best_acc:
                    best_acc = acc
                    #model.save(name_new)
                    count = 0
                else:
                    count += 1
                    
                if count == 20:
                    lr = lr * .5
                    optimizer = SAM(Adam(learning_rate=lr))
                    #optimizer = Adam(learning_rate=lr)
                    count = 0
                
                #if acc >= .87:
                    #beta = 0
                #else:
                    #beta = 1
                    
                if acc >= .65:
                    if sr < best_sr:
                        best_sr = sr
                        #model.save(name)
            step +=1   
            if step >= n_steps:
                break   

def save_images(X, type):
    
    for i in range(0,10):
        image = X[i]
        image = image * 255.0
        image = image.reshape(32,32,3)
        image = np.array(image, dtype=np.uint8)
        cv2.imwrite('images/'+type +str(i)+'.png', image)
        
        

def save_images(X, type):
    
    
    image = X
    image = image * 255.0
    image = image.reshape(32,32,3)
    image = np.array(image, dtype=np.uint8)
    cv2.imwrite('images/'+type +str(i)+'.png', image)
    

                          
train() 
#model.save('vanilla_model')



    

