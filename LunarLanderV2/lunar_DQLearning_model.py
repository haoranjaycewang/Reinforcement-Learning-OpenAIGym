from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam

class Q_Model():
    """
    Q Network/function used to calculate the action-values Q(s,a)
    """
    def __init__(self,action_space,state_space,hidden_nodes,layers,learning_rate):
        self.action_space = action_space
        self.state_space = state_space
        self.nodes = hidden_nodes
        self.layers = layers
        self.learning_rate = learning_rate

    def build(self):
        model = Sequential()
        #Input Layer
        model.add(Dense(self.nodes, input_dim=self.state_space,activation=relu))
        for _ in range(self.layers):
            model.add(Dense(self.nodes, activation=relu))

        #output layer
        model.add(Dense(self.action_space, activation=linear))
        model.compile(
                      optimizer=Adam(lr=self.learning_rate),
                      loss='mse',
                      metrics=['accuracy'])
        model.summary()


        return model




        return model
