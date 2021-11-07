import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
# new line
import gym
from gym import spaces
from gym.utils import seeding
import skvideo.io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import random
from datetime import datetime

class DQN:
    def __init__(self, env):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0
        self.state_size = env.observation_space.shape[0]

        #######################
        # Change these parameters to improve performance
        self.density_first_layer = 128
        self.density_second_layer = 128
        self.num_epochs = 1
        self.batch_size = 64
        self.epsilon_min = 0.1

        # epsilon will randomly choose the next action as either
        # a random action, or the highest scoring predicted action
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.99

        # Learning rate
        self.lr = 0.0001

        #######################

        self.rewards_list = []

        self.replay_memory_buffer = deque(maxlen=900000)
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]


        self.policy_model = self.initialize_model()
        self.target_model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(self.density_first_layer, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(self.density_second_layer, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))

        # Compile the model
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):

        # The epsilon parameter decides whether we are using the 
        # Q-function to determine our next action 
        # or take a random sample of the action space. 
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)

        # Get a list of predictions based on the current state
        predicted_actions = self.policy_model.predict(state)

        # Return the maximum-reward action
        return np.argmax(predicted_actions[0])

    def add_to_replay_memory(self, state, action, reward, next_state, done, next_action):
        self.replay_memory_buffer.append((state, action, reward, next_state, done, next_action))

    def update_target_model(self):
        self.target_model.set_weights(self.policy_model.get_weights())
        return self.target_model
    
    def learn(self):
        cur_batch_size = min(len(self.replay_memory_buffer), self.batch_size)
        #mini_batch = random.sample(self.replay_memory_buffer, cur_batch_size)
        mini_batch = self.get_random_sample_from_replay_mem()
        
        # batch data
        states = np.ndarray(shape = (cur_batch_size, self.state_size)) 
        actions = np.ndarray(shape = (cur_batch_size, 1))
        rewards = np.ndarray(shape = (cur_batch_size, 1))
        next_states = np.ndarray(shape = (cur_batch_size, self.state_size))
        dones = np.ndarray(shape = (cur_batch_size, 1))
        next_actions = np.ndarray(shape = (cur_batch_size, 1))

        temp=0
        for exp in mini_batch:
            states[temp] = exp[0]
            actions[temp] = exp[1]
            rewards[temp] = exp[2]
            next_states[temp] = exp[3]
            dones[temp] = exp[4]
            next_actions[temp] = exp[5]
            temp += 1
        
         
        qhat_next = self.target_model.predict(next_states)
        
        # set all Q values terminal states to 0
        qhat_next = qhat_next * (np.ones(shape = dones.shape) - dones)
        # choose max action for each state
        #sample_qhat_next = np.max(sample_qhat_next, axis=1)
        
        qhat = self.policy_model.predict(states)
        
        for i in range(cur_batch_size):
            a = actions[i,0]
            b = next_actions[i,0]
            qhat[i,int(a)] = rewards[i] + self.gamma * qhat_next[i, int(b)]
            
        q_target = qhat
            
        self.policy_model.fit(states, q_target, epochs = 1, verbose = 0)
    
    

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        next_actions = np.array([i[5] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list, next_actions

    # Get a batch_size sample of previous iterations
    def get_random_sample_from_replay_mem(self):
        cur_batch_size = min(len(self.replay_memory_buffer), self.batch_size)
        random_sample = random.sample(self.replay_memory_buffer, cur_batch_size)
        #random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    # Run the keras predict using the current state as input.
    # This will choose the next step.
    def predict(self, current_state):
        return self.policy_model.predict(current_state)

    def train(self, num_episodes=2000, can_stop=True, writer=None):

        frames = []

        for episode in range(num_episodes):

            # state is a vector of 8 values:
            # x and y position
            # x and y velocity
            # lander angle and angular velocity
            # boolean for left leg contact with ground
            # boolean for right leg contact with ground
            state = self.env.reset()
            reward_for_episode = 0
            done = False
            state = np.reshape(state, [1, self.num_observation_space])
            
            # use epsilon decay to choose the next state
            current_action = self.get_action(state)
            while not done:

                if episode % 50 == 0:
                    frame = env.render(mode='rgb_array')

                if episode % 50 == 0:
                    frames.append(frame)                    


                
                next_state, reward, done, info = self.env.step(current_action)

                # Reshape the next_state array to match the size of the observation space
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                
                # use epsilon decay to choose the next state
                next_action = self.get_action(next_state)

                # Store the experience in replay memory
                self.add_to_replay_memory(state, current_action, reward, next_state, done, next_action)

                # add up rewards
                reward_for_episode += reward
                state = next_state
                current_action = next_action
                self.update_counter()

                # update the model
                #self.learn_and_update_weights_by_reply()
                self.learn()

                #if done:
                #    break
            self.rewards_list.append(reward_for_episode)
            
            self.update_target_model()

            # Create a video from every 10th episode
            if episode % 50 == 0:
                fname = "./tmp/videos/episode"+str(episode)+".mp4"
                skvideo.io.vwrite(fname, np.array(frames))
                del frames
                frames = []

            # Decay the epsilon after each experience completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                #self.epsilon *= min(0.995,(self.epsilon_decay + counter*(0.000075)))
               
                
            # Check for breaking condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])

            # Once the mean average of rewards is over 200, we can stop training
            if last_rewards_mean > 250 and can_stop:
                print("DQN Training Complete...")
                break
            #print(episode, "\t: Episode || Reward: ",reward_for_episode, "\t|| Average Reward: ",last_rewards_mean, "\t epsilon: ", self.epsilon )
            print("Episode: {}, Reward: {:.3f}, Avg Reward :{:.3f}, Epsilon:{:.4f}".format(episode, reward_for_episode,last_rewards_mean, self.epsilon))
            
            if(writer != None):
               with writer.as_default():
                   tf.summary.scalar('Reward/Train', reward_for_episode, episode)
                   tf.summary.scalar('Avg-Reward/Train', last_rewards_mean, episode)
               
        if(writer != None):      
           tf.summary.flush(writer=writer)
            
        

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, name):
        self.model.save(name)


if __name__=="__main__":
    rewards_list = []

    # Run 100 episodes to generate the initial training data
    #num_test_episode = 100

    # Create the OpenAI Gym Enironment with LunarLander-v2
    env = gym.make("LunarLander-v2")

    # set the numpy random number generatorseeds
    env.seed(21)
    np.random.seed(21)

    # max number of training episodes
    training_episodes = 1000

    # number of test runs with a satisfactory number of good landings
    #high_score = 0
 
    # initialize the Deep-Q Network model
    model = DQN(env)
    
    writer = tf.summary.create_file_writer("./logs/sarsa_lunar_landing")

    # Train the model
    model.train(training_episodes, True, writer=writer)

    print("Starting Testing of the trained model...")

    # Run 100 episodes to generate the initial training data
    num_test_episode = 100

    # number of test runs with a satisfactory number of good landings
    high_score = 0

    # Create the OpenAI Gym Enironment with LunarLander-v2
    #env = gym.make("LunarLander-v2")
    rewards_list = []

    #model = load_model("final_model.h5")
    done = False
    frames = []

    # Run some test episodes to see how well our model performs
    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        reward_for_episode = 0
        done = False
        while not done:

            if test_episode % 50 == 0:
                frame = env.render(mode='rgb_array')
                frames.append(frame)

            selected_action = np.argmax(model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode += reward
        rewards_list.append(reward_for_episode)
        print(test_episode, "\t: Episode || Reward: ", reward_for_episode)
        if reward_for_episode >= 200:
            high_score += 1
        if test_episode % 50 == 0:
            fname = "./tmp/videos/testing_run"+str(test_episode)+".mp4"
            skvideo.io.vwrite(fname, np.array(frames))
            del frames
            frames = []
            
        rewards_mean = np.mean(rewards_list[-100:])    
        with writer.as_default():    
            tf.summary.scalar('Reward/Test', reward_for_episode, test_episode)
            tf.summary.scalar('Avg-Reward/Test', rewards_mean, test_episode)
       
    tf.summary.flush(writer=writer)        
           

    
    print("Average Reward: ", rewards_mean )
    print("Total tests above 200: ", high_score)
    model.policy_model.save('mymodel.h5')       
