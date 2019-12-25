import gym,time
from algorithms.DQN import DQN
#env = gym.make('AirRaid-v0')
env = gym.make('Enduro-v0')
#o = env.observation_space
brain = DQN([210,160,3],9)
for i_episode in range(10000):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = brain.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        brain.store_exp(observation,action,observation_,reward,done)
        observation = observation_
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()