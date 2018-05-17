# DRL-ice-hockey

The repository contains the codes about the network structure of [Deep Reinforcement Learning in Ice Hockey
for Context-Aware Player Evaluation](inprogress).  

***Network Structure***:  

| name        | nodes           | activation function  |
| ------------- |:-------------:| -----:|
| LSTM Layer    | 512           | N/A |
| Fully Connected Layer 1| 1024     |  Relu |
| Fully Connected Layer 2| 1000      |  Relu |
| Fully Connected Layer 3| 3      |  N/A |

***Image of network structure***:  
<img src=./images/DP-lstm-model-structure.png alt="drawing" style="width:50px;"/>

<!---![model-structure](./images/DP-lstm-model-structure.png =250x250)--->

***Training method***  
We are using the on-policy prediction method [Sarsa](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action) (State–Action–Reward–State–Action).
It's a Temporal Difference learning method, and estimate the player performance by Q(s,a), where state s is a series of game contexts and action a is the motion of player.

***Running:***  
The origin works uses a private play-by-play dataset from [Sportlogiq](http://sportlogiq.com/en/), which has not been published.

If you want to run the network, please prepare your won sequential dataset, please organize the data according to network input in the format of Numpy.   

***Package required:***  
1. Numpy 
2. Tensorflow

we are still updating this repository.
