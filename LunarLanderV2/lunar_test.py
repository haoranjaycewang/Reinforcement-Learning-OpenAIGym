import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
env = gym.make('LunarLander-v2')
action_space = env.action_space.n
state_space = env.observation_space.shape[0]

lunar_agent = tf.keras.models.load_model('C:/Users/haora/gymEnv/LunarLand/models/lunar_model_score215.65755254109038.h5')
file_name = 'lunarLand_test_data.txt'
datafile = open(file_name,"w+")
episodes = 10
lunar_agent.summary()
#print(lunar_agent.get_weights())

for e in range(episodes):
    state = env.reset()
    accum_reward = 0
    while True:
        env.render()
        state = np.reshape(state,[1,state_space])
        prediction = lunar_agent.predict(state)
        action = np.argmax(prediction[0])
        next_state, reward, done, _ = env.step(action)
        accum_reward += reward

        if done:
            break
    print("episode:{}/{} | score:{}".format(e,episodes,accum_reward))
    datafile.write(str(e)+','+str(accum_reward)+'\n')

env.close()
datafile.close()

x, y = np.loadtxt(file_name, delimiter=',', unpack=True)
plt.plot(x,y, label='Loaded from file!')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Episode VS Score Test Trend: 128-128-128')
plt.legend()
plt.show()
