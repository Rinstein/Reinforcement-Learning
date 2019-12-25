import gym,time
from algorithms.DQN import DQN
#env = gym.make('AirRaid-v0')
#env = gym.make('Breakout-v0')
env = gym.make('CartPole-v0')

#获取观测的尺寸
observation_space = env.observation_space.shape
#获取动作的个数，在gym中，动作均是有数字指代
action_space = env.action_space.n

brain = DQN(observation_space,action_space,env_name=str(env.env),net_mode='fc')
for i_episode in range(10000):
    observation = env.reset()
    sum_r = 0
    for t in range(10000):
        #time.sleep(1)
        env.render()
        #action = env.action_space.sample()
        action = brain.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        brain.store_exp(observation,action,observation_,reward,done)
        observation = observation_
        sum_r = sum_r + reward
        #print('#t= '+str(t))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print('#epoch= '+str(i_episode)+' #sum_reward= '+str(sum_r))
            break
env.close()