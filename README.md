# sample-map-routing
This is an example of a reinforcement learning (RL) AI to generate map routes as a sequential decision-making task where the agent learns optimal paths through interaction with a map-based environment.

## Setup
```
conda create -n map_rl_env python=3.10
conda activate map_rl_env
conda install pip
pip install osmnx networkx numpy pygame matplotlib
pip install mercantile contextily pillow
pip install scipy tensorflow
```

Now in VS Code open the Command Palette (Ctrl+Shift+P or ⌘+Shift+P) and search for Python: Select Interpreter.  The search for the map_rl_env environment.  Copy that and replace the path in .vscode/settings.json.  i.e.
```
{
  "python.pythonPath": "/opt/anaconda3/envs/map_rl_env/bin/python"
}
```

Then open a terminal and execute below to confirm the python environment
```
which python
```

## Execution
There's also an offline trainer that will run through 100 episodes and then update the stored model at ```./models/dqn_model.weights.h5```.  You can launch the offile trainer with ```python train_offline.py```.  It's been trained with 100 episodes but you'll need to train it for at least 1,000.  if you decide to update the state, i.e. adding a traffic indicator, then you'll need to retrain your model from scratch.  Also, any new state you add to your model should include an update to the reward calculation in the step function of your GraphEnv environment that is part of the train_offline.py program.

There's an interactive program that you can launch with ```python main.py```.  This will birng up a window with a preloaded map enviornment.  You select the starting point and destination by clicking on the map.  By default the agent uses your stored model to move from a starting position to the goal.  You can switch to follow a shortest path algorithm by hitting the ```R``` key.  You can also hide or display he mini-map by hitting the ```M``` key.


## Notes
1) You can adjust the model architecture and hyperparameters to improve you AI.  See the _build_model method of the ./ai/dqn_agent.py file for more information.  If yu train a different model you probably want to rename the current model file in ./models/dqn_model.weights.h5 so that you don't overwrite the existing trained data.

2) And then in the step function of the ./train_offline.py program you'll find the reward system that is used to inform the agent when it makes good or bad decisions.  You can adjust this to alter the behavior of your agent, but remember that you may not notice a difference until you've run a thousand episodes with the changes.

3) The ./main.py program uses the agent you've trained to determine the best path from a starting point to a goal.  The interactive mao loads with a default address that we use to train the agent, a simple grid, but you can chnage the address to load a different map and the agent should work the same.  You may want to make this an input into the program that the user can choose.