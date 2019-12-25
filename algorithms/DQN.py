from keras import models,layers,optimizers
import random
import numpy as np
from copy import copy

class DQN():
    def __init__(self,input_dimension,output_dimension,learning_rate=0.01,reward_decay=0.9,
                 epsilon_greedy=0.1,):
        #基本参数
        self.input_space = input_dimension
        self.output_space = output_dimension
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon_greedy

        #经验池参数
        self.exp_max_size = 5000
        self.exp_count = 0
        self.exp_pool = [None]*self.exp_max_size
        self.train_batch = 32
        self.train_step = 0

        #Q网络
        self.q_net = self.build_net()
        self.q_net.summary()
        #复制模型 结构
        self.target_net = models.clone_model(self.q_net)
        #复制模型 参数
        self.target_net.set_weights(self.q_net.get_weights())

    def choose_action(self,state):
        t = random.random()
        if(t<self.epsilon):
            return random.randint(0,self.output_space-1)
        else:
            q_value = self.q_net.predict([[state]])[0]
            action = np.argmax(q_value)
            return action

    def build_net(self):
        #这里首先构建简单的卷积神经网络
        model = models.Sequential()
        model.add(layers.Conv2D(16,(3,3),input_shape=self.input_space))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(16,(3,3)))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.output_space))

        sgd = optimizers.SGD(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=sgd,metrics=['mse'])
        return model

    def store_exp(self,state,action,state_,reward,done):
        self.exp_pool[self.exp_count%self.exp_max_size] = [state,action,state_,reward,done]
        self.exp_count = self.exp_count+1

        if self.exp_count>500 and self.exp_count%20==0:
            self.train()
            print('#train step: '+str(self.train_step))

    def train(self):
        #构造数据集
        t = self.exp_count
        if self.exp_max_size<self.exp_count:
            t = self.exp_max_size
        dataset = random.sample(self.exp_pool[0:t],self.train_batch)
        input_data = []
        #output_data = []
        yi = []
        for t in dataset:
            input_data.append(t[0])

            #计算更新yi
            r = t[3]
            if not t[4]:
                r = r + self.gamma*max(self.target_net.predict([[t[0]]])[0])
            y = self.q_net.predict([[t[0]]])[0]
            #output_data.append(copy(y))
            y[t[1]] = r
            yi.append(y)
        self.q_net.train_on_batch([input_data],[yi])
        self.train_step = self.train_step + 1
        print('#train step:'+str(self.train_step))
        #每训练100轮进行替换
        if self.train_step%200 == 0:
            print('替换target网络')
            self.target_net.set_weights(self.q_net.get_weights())





