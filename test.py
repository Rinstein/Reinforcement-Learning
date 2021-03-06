import gym,time
from algorithms.DQN import DQN
#env = gym.make('AirRaid-v0')
#env = gym.make('Breakout-v0')
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')

#获取观测的尺寸
observation_space = env.observation_space.shape
#获取动作的个数，在gym中，动作均是有数字指代
action_space = env.action_space.n

brain = DQN(observation_space,action_space,env_name=str(env.env),net_mode='fc')
brain.load_model()

TEST = 100
i_episode = 0
while True:
    #嵌入阶段测试代码
    if i_episode%100==0:
        t_sum_reward = 0
        for i in range(TEST):
            observation = env.reset()
            while True:
                # action = env.action_space.sample()
                action = brain.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                observation = observation_
                t_sum_reward = t_sum_reward + reward
                if done:
                    break
        print('#episode=',i_episode,' #average reward=',t_sum_reward/TEST)
    i_episode = i_episode + 1
    observation = env.reset()
    sum_r = 0
    t = 0
    while True:
        t = t + 1
        #time.sleep(1)
        env.render()
        #action = env.action_space.sample()
        action = brain.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        #print('返回的信息为', observation_,reward,done,info)
        brain.store_exp(observation,action,observation_,reward,done)
        observation = observation_
        sum_r = sum_r + reward
        #print('#t= '+str(t))
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            # print('#epoch= '+str(i_episode)+' #sum_reward= '+str(sum_r))
            break
env.close()