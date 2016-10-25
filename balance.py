#genetic approach

import sys
import gym
import numpy as np

weights = np.random.normal(0.0,0.1,4)
memory = []

if len(sys.argv) == 3:
	seed = int(sys.argv[0])
	episodes = int(sys.argv[1])
	episode_length = int(sys.argv[2])
else:
	seed = 10
	episodes = 10
	episode_length = 300
learning_rate = 0.1

def sigmoid(x):
    return 1/(1+np.exp(x))

def length(a):
    x = 0
    for element in a:
        x = x + element
    return x / len(a)

def execute(observation, weights):
    return int(round(sigmoid(length(observation*weights))))

def sort(memory):
    old = len(memory)
    mean = np.sum(np.transpose(np.array(memory))[1]) / len(memory)
    i=0
    while i < len(memory):
        if memory[i][1] < mean:
            memory.pop(i)
        i += 1
    print("killed " + str(old-len(memory))) 

env = gym.make('CartPole-v0')
for idx in range(episodes):
    learning_rate *= 1 - 1/2/episodes
    for i_episode in range(seed):
        observation = env.reset()
        if idx == 0:
            weights = np.random.normal(0,0.2,4)
        else:
            weights = memory[i_episode][0] + np.random.normal(0,learning_rate,4)
        for t in range(episode_length):
            env.render()
            action = execute(observation,weights)
            observation, reward, done, info = env.step(action)
            if(done):
                print("Episode finished after {} timesteps".format(t))
                memory.append((weights, t))
                break
    sort(memory)
    seed = len(memory)

print(memory[np.argmax(np.transpose(memory)[1])])
