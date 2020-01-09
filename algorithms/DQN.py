from keras import models,layers,optimizers
import random,time
import numpy as np

class DQN():
    def __init__(self,input_dimension,output_dimension,learning_rate=0.0001,reward_decay=0.9,
                 epsilon_greedy=0.1,env_name='',net_mode='fc'):
        #基本参数
        self.input_space = input_dimension
        self.output_space = output_dimension
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon_greedy
        self.env_name = env_name.replace('<','').replace('>','')
        self.net_mode = net_mode

        #经验池参数
        self.exp_max_size = 10000
        self.exp_count = 0
        self.exp_pool = [None]*self.exp_max_size

        #训练参数
        self.train_start_step = 1000
        self.train_batch = 64
        self.train_step = 0
        self.train_interval = 300
        self.train_replace_target = 100

        #Q网络
        if net_mode == 'fc':
            self.q_net = self.build_net_fc()
        elif net_mode == 'conv':
            self.q_net = self.build_net_conv()
        else:
            print('神经网络模型只能是fc/conv，卷积或者全连接，输入net_mode参数有误')
            exit(0)
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

    def build_net_conv(self):
        #这里首先构建简单的卷积神经网络
        model = models.Sequential()
        model.add(layers.Conv2D(16,(3,3),input_shape=self.input_space))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
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
        model.compile(loss='mean_squared_error',optimizer=sgd)
        return model

    def build_net_fc(self):
        #这里首先构建简单的卷积神经网络
        model = models.Sequential()
        model.add(layers.Dense(32,activation='relu',input_shape=self.input_space))
        #model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(32,activation='relu'))
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(self.output_space))

        sgd = optimizers.SGD(lr=self.learning_rate)
        model.compile(loss='mean_squared_error',optimizer=sgd)
        return model

    def store_exp(self,state,action,state_,reward,done):
        #注意进行归一化处理
        if self.net_mode == 'conv':
            self.exp_pool[self.exp_count%self.exp_max_size] = [np.array(state)/255,action,np.array(state_)/255,reward,done]
        else:
            self.exp_pool[self.exp_count%self.exp_max_size] = [np.array(state),action,np.array(state_),reward,done]
        self.exp_count = self.exp_count+1

        if (self.exp_count>self.train_start_step) and (self.exp_count%self.train_interval==0):
            self.train()
            print('#train step= '+str(self.train_step))
            #print('#pool_size= '+str(self.exp_count))

    def train(self):
        #构造数据集
        t = self.exp_count
        if self.exp_max_size<self.exp_count:
            t = self.exp_max_size
        dataset = random.sample(self.exp_pool[0:t],self.train_batch)
        input_data = []
        yi = []
        for t in dataset:
            input_data.append(t[0])
            #计算更新yi
            r = t[3]
            if not t[4]:
                r = r + self.gamma*max(self.target_net.predict([[t[2]]])[0])
            y = self.q_net.predict([[t[0]]])[0]
            y[t[1]] = r
            yi.append(y)

        #计算网络性能
        # y = self.q_net.predict([[dataset[0][0]]])[0]
        # print(y)
        # r = dataset[0][3]
        # if not dataset[0][4]:
        #     r = r + self.gamma * max(self.target_net.predict([[dataset[0][0]]])[0])
        # y[dataset[0][1]] = r
        # print(y)

        #metrics = self.q_net.train_on_batch([input_data],[yi])
        metrics = self.q_net.fit([input_data],[yi],epochs=1)
        #print(input_data,'----',yi)

        #if self.train_step % 20 == 0:
        print("  loss= "+str(metrics))
        # time.sleep(2)
        self.train_step = self.train_step + 1
        #一段训练时间进行网络参数替换
        if self.train_step%self.train_replace_target == 0:
            print('保存当前网络，并替换target网络')
            self.save_model()
            self.target_net.set_weights(self.q_net.get_weights())

    #模型的默认保存加载位置 同级目录
    def save_model(self):
        self.q_net.save('./models/'+self.env_name+'_q_net.h5','w')
        self.target_net.save('./models/'+self.env_name+'_target_net.h5')

    def load_model(self):
        self.q_net = models.load_model('./models/'+self.env_name+'_q_net.h5')
        self.target_net = models.load_model('./models/'+self.env_name+'_target_net.h5')
        print('载入模型成功')





