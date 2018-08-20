# DRL-ice-hockey

The repository contains the codes about the network structure of paper "[Deep Reinforcement Learning in Ice Hockey
for Context-Aware Player Evaluation](https://www.ijcai.org/proceedings/2018/0478.pdf)".  

## Network Structure:  

| name        | nodes           | activation function  |
| ------------- |:-------------:| -----:|
| LSTM Layer    | 512           | N/A |
| Fully Connected Layer 1| 1024     |  Relu |
| Fully Connected Layer 2| 1000      |  Relu |
| Fully Connected Layer 3| 3      |  N/A |

## Image of network structure:  

<img src=./images/DP-lstm-model-structure.png alt="drawing" height="320" width="420"/>

<!---![model-structure](./images/DP-lstm-model-structure.png =250x250)--->

## Training method 
We are using the on-policy prediction method [Sarsa](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) (State–Action–Reward–State–Action).
It's a Temporal Difference learning method, and estimate the player performance by Q(s,a), where state s is a series of game contexts and action a is the motion of player.

## Running:  
Use ```python td_three_prediction_lstm.py``` to train the neural network, which produce the Q values. Goal-Impact-Metric is the different between consecutive Q values.  
The origin works uses a private play-by-play dataset from [Sportlogiq](http://sportlogiq.com/en/), which is not allowed to publish. 

### About the input: 
If you want to run the network, please prepare your won sequential dataset, please organize the data according to network input in the format of Numpy. As it's shown in ```td_three_prediction_lstm.py```, the neural network requires three input files: 

* reward
* state_input (conrtains both state features and one hot represetation of action) 
* state_trace_length

Each input file must has the same number of rows _R_ (corresponding to number of events in a game). In our paper, we have trace length equals to 10, so reward is an _R_\*10 array, state_input is an _R_\*10\*feature_number array and state_trace_length is an one demensional vector that tells the length of plays in a game.

To be specific, if you want to directly run this python RNN scripy, you need to prepare the input in this way. In each game file, there are three .mat files representing reward, state_input and state_trace_length.
 
 - **GameDirectory_xxx**
   - *dynamic_rnn_reward_xxx.mat*
   - *dynamic_rnn_input_xxx.mat*
   - *hybrid_trace_length_xxx.mat*
 
 in which *xxx* is a random string.

The data must be ***standardized or normalized*** before inputing to the neural network, we are using the ***sklearn.preprocessing.scale*** 

## Package required:
1. Numpy 
2. Tensorflow
3. Scipy
4. Matplotlib
5. scikit-learn

## LICENSE:
MIT LICENSE

we are still updating this repository.
