import os
import pickle
import neat
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np

channel = EngineConfigurationChannel()

env = UE(file_name='unity/test_scene/New Unity Project', seed=100, side_channels=[channel])

channel.set_configuration_parameters(time_scale=6)

#load winner
with open('result/best_genome', 'rb') as f:
    c = pickle.load(f)

print("load genome:")
print(c)

#load config file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)

env.reset()

behavior_name = env.get_behavior_names()[0]

spec = env.get_behavior_spec(behavior_name)

while True:
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    done = False
    step_count = 0
    while not done:
        actions = np.zeros(shape=(1, 2))
        if len(decision_steps) > 0:
            action = net.activate(decision_steps[0].obs[0])
            actions[0] = action

        if len(decision_steps.agent_id) > 0:
            env.set_actions(behavior_name, actions)

        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        step_count += 1
        if step_count > 200:  # time out
            done = True
            env.reset()
