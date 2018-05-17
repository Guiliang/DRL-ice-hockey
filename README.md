# DRL-ice-hockey

The repository contains the codes about the network structure of [Deep Reinforcement Learning in Ice Hockey
for Context-Aware Player Evaluation](inprogress).  

Network Structure:

| name        | nodes           | activation function  |
| ------------- |:-------------:| -----:|
| LSTM Layer    | 512           | N/A |
| Fully Connected Layer 1| 1024     |  Relu |
| Fully Connected Layer 2| 1000      |  Relu |
| Fully Connected Layer 3| 3      |  N/A |

The image structure is:
<img src=./images/DP-lstm-model-structure.png alt="drawing" style="width: 200px;"/>

<!---![model-structure](./images/DP-lstm-model-structure.png =250x250)--->

./images/DP-lstm-model-structure.png
If you want to run it, please organize the data according to network input in array format of Numpy.  
we are still updating it.  

Package required:
1. Numpy 
2. Tensorflow
