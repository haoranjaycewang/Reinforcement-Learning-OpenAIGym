import gym
import numpy as np
from collections import deque
from lunar_DQLearning_model import *
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import tensorflow as tf
'''
Following the Pseudo-Algorithm from Artem Oppermann's Article on TowardDataScience
https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47
'''
directory = 'C:/Users/haora/gymEnv/LunarLand/models/'
env = gym.make('LunarLander-v2')
file_name = 'lunarLand_datafile.txt'
datafile = open(file_name,"w+")
episodes = 500

mini_batch_size = 32
hidden_layers = 2

''' Hyperparameters'''
batch_size = 64
gamma = 0.98
epsilon_decay = 0.98
epsilon_min = 0.02
memory_size = 10000

action_space = env.action_space.n
state_space = env.observation_space.shape[0]

q_setup = Q_Model(action_space,state_space,128,hidden_layers,0.001)
q_network = q_setup.build()
reward_record = []
state = env.reset()
#Initialize replay memory
replay_memory = deque(maxlen=memory_size)

for e in range(episodes):
    #Initialize state s_t
    accum_reward = 0
    if 1/(math.sqrt(e+1))>epsilon_min: #From Eq12 from article
        epsilon = 1/(math.sqrt(e+1))
    else:
        epsilon = epsilon_min  #minimum epsilon

    state = env.reset()

    while True:
        #With a probability epsilon select a random action a_t
        p = np.random.uniform(0,1)
        if p <= epsilon:
            action = env.action_space.sample()
        else:
            #Action_space = [Nothing,fire left, fire main, fire right]
            state = np.reshape(state,[1,state_space])
            prediction = q_network.predict(state)
            #Taking the index of highest value because we want a discrete number for the action
            action = np.argmax(prediction[0])
        #Execute action and observe reward and state s_t+1
        next_state, reward, done, _ = env.step(action)
        accum_reward += reward
        #Store Transitions (state,action,reward,next_state) in memory
        replay_memory.append((state,action,reward,next_state,done))
        #Set s_t+1 = s_t
        state = next_state
        #Sample random minibatch of transitions (state,action,reward,next_state) from memory
        if len(replay_memory) > 64:
            sample_batch = random.sample(replay_memory,batch_size)

            #Vector of properties of sample batch
            states = np.array([np.reshape(i[0],[1,state_space]) for i in sample_batch])
            actions = np.array([i[1] for i in sample_batch])
            rewards = np.array([i[2] for i in sample_batch])
            next_states = np.array([np.reshape(i[3],[1,state_space]) for i in sample_batch])
            dones = np.array([i[4] for i in sample_batch])
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            current_state_Q = q_network.predict_on_batch(states)
            # Predict Optimal Q function using target network predictions
            next_state_Q = rewards + gamma*np.amax(q_network.predict_on_batch(next_states), axis=1)*(1-dones)
                        # Create index from batch size
            indices = np.array([i for i in range(batch_size)])
            # Map actions to optimal action-state value function
            current_state_Q[[indices], [actions]] = next_state_Q
            # Train model with states and predicted action-state value function
            #target_network.set_weights(q_network.get_weights())
            q_network.fit(states, current_state_Q, epochs=1, verbose=0)
            #Set the target Network's weights as theta_i-1 since its the weight of the q_network at a previous timestep
        if done:
            break
    print("episode:{}/{} | score:{} | epsilon:{}".format(e,episodes,accum_reward,epsilon))
    reward_record.append(accum_reward)
    if len(reward_record)>100 and np.mean(reward_record[-100:]) >= 200:
        q_network.save(directory+"lunar_model_score{}.h5".format(accum_reward))
        break
    datafile.write(str(e)+','+str(accum_reward)+'\n')

env.close()
datafile.close()

# Plots Episode VS Score of the training session, can comment out if unwanted
x, y = np.loadtxt(file_name, delimiter=',', unpack=True)
plt.plot(x,y, label='Loaded from file!')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Episode VS Score Trend: 128-128-128|e_decay:{}|e_min:{}|gamma:{}|batchsize:{}'.format(epsilon_decay,epsilon_min,gamma,batch_size))
plt.legend()
plt.show()
