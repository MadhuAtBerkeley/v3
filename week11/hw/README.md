# Homework 11 -- Fun with OpenAI Gym!


# Assignment
In this homework, you will be training a Lunar Lander module to land properly **using your Xavier NX**. There is a video component to this file, so use a display or VNC.

## DQN Model 
The config of deep neural network is given below


```
        self.density_first_layer = 128
        self.density_second_layer = 128
        self.num_epochs = 1
        self.batch_size = 64
        self.epsilon_min = 0.1
```



# Results 
## Logs
 * Train logs : In the file
 * Test logs : In the file
 * test log output is as below


```
DQN Training Complete...

.
.
.
Average Reward:  243.09916996497867
```

## Videos

* First episode : https://w251-s3-bucket.s3.us-west-2.amazonaws.com/HW11_videos/episode0.mp4
* Intermediate : https://w251-s3-bucket.s3.us-west-2.amazonaws.com/HW11_videos/episode50.mp4
* Best : https://w251-s3-bucket.s3.us-west-2.amazonaws.com/HW11_videos/episode500.mp4
* Last : https://w251-s3-bucket.s3.us-west-2.amazonaws.com/HW11_videos/episode550.mp4
* Test0 : https://w251-s3-bucket.s3.us-west-2.amazonaws.com/HW11_videos/testing_run0.mp4
* Test1: https://w251-s3-bucket.s3.us-west-2.amazonaws.com/HW11_videos/testing_run50.mp4

## Tensorboard Plots

* DQN plots : https://tensorboard.dev/experiment/uD868PK7QeynmPWyomD8tQ/#scalars
* SARA Plots : https://tensorboard.dev/experiment/zdznv1lUQxWqqx0jP7x8pA/#scalars

# Questions
Submit a write-up of the tweaks you made to the model and the effect they had on the results. 
Questions to answer:
1) What parameters did you change, and what values did you use? 
Ans:  I tried changing following parameters
 *  Number of neurons in the hidden layers.  There was an improvement in the acg utility values with larger hidden layer size (32, 64 and 128).  The functional approximation of Q function improves with higher number of neurons - especially when the number of states are high. 
 *  Changed the learning rate to 0.0001. The learning rate set lower as robot makes high explorations initially.
 *  I tried batch_size of 32, 64 and 128. Higher batch sizes improve the quality of gradients (by reducing the gradient noise variance) and reduce computational time due to parallelization on GPUs.  However, larger batch sizes have slow convergence and might get stuck in local minima. The batch_size should be large enough for higher gradient accuracy/low computational speed and small enough for generalization/regularization. I chose batch_size of 64 as the performance was better than 32.


2) Did you try any other changes (like adding layers or changing the epsilon value) that made things better or worse?

Ans: I found vanilla DQN took more time for converegnce.  Hence, I modified the code to implement 
 *  Double DQN : Used separate Deep networks for target and policy estimation.
 *  SARSA : I tried SARSA as well but Q learning performed better than SARSA.  SARSA is an on-policy training and Q training is off-policy
 *  The epsilon_decay is used to change epsilon value and I did not change initial epsilon value or default epsilon decay


3) Did your changes improve or degrade the model? How close did you get to a test run with 100% of the scores above 200?

Ans: Increasing epsilon_decay value higher than 0.9999 would increase exploration (reduced learning) even after several episodes. Lowering epsilon_decay below 0.99 decreased exploration and aggressive learning with incorrect Q estimates in the begining. The default value of 0.995 worked best for me.

Increasing learning rate higher than 0.001 resulted in aggressive learning with Q estimates with high bias and reduced the performance.

I got 76 out of 100 test episodes above 200

4) Based on what you observed, what conclusions can you draw about the different parameters and their values? 

Ans: Reinforcement learning is sensitive to hyperparameter and requires right amount exploration and exploitation to train up.

5) What is the purpose of the epsilon value?

Ans: The epsilon value determines exploration of the agent model free Q-learning. The Q-learning uses epislon-greedy algorithm to decide next actions and explore unseen states. At the begining, the agent has a very minimal understanding of the environment and typically uses high explorartion. As it runs through many episodes, the epsilon value is reduced to allow learning from the improved understanding of the environment.

6) Describe "Q-Learning".

Ans: Q-learning is off-policy version of TD-learning (temporal difference).  It allows model-free reinceforcement learning of MDP. Q-learning uses greedy-epsilon algorithm for choosing next actions. It directly learns the Q value (average utility of MDP states) using a recursion where estimates of Q(s,a) are iteratively improved by selecting an off-policy action that maximizes the reward + average utility of the next state (after taking the action). Q learning is different than SARSA - SARSA is an on-policy algorithm.

## Grading is based on the changes made and the observed output, not on the accuracy of the model.

We will compare results in class. The biggest Average Reward after the test run "wins":

```
Average Reward:  243.09916996497867
```

# Hint: you can disable video output to speed up the training process.
