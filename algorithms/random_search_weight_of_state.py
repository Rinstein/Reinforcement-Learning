import gym
import numpy as np

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(300):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    bestparams = None
    bestreward = 0
    for _ in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                print(bestparams,bestreward)
                r = 0
                for __ in range(1000):
                    #bestparams = [-0.13023735 ,0.22199065 ,0.75564959 ,0.34933432]
                    tr = run_episode(env,bestparams)
                    r = r+tr
                print('1000场测试平均reward=',r/1000)
                break
