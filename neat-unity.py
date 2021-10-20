import numpy as np
import neat
import pickle
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import visualize
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

channel = EngineConfigurationChannel()

env = UE(file_name='Unity/test_18/New Unity Project', seed=1, side_channels=[channel])

channel.set_configuration_parameters(time_scale=20)

env.reset()

behavior_name = env.get_behavior_names()[0]

spec = env.get_behavior_spec(behavior_name)

print("Number of observations : ", spec.observation_shapes[0][0])

if spec.is_action_continuous():
    print("The action is continuous")

if spec.is_action_discrete():
    print("The action is discrete")

print(env.get_behavior_names(), behavior_name, 'name')

generation = 0


def run_agent(genomes, config):
    # Init NEAT
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    # print(decision_steps.agent_id)
    # print(decision_steps.obs[0])
    nets = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

    # Main loop
    global generation
    generation += 1

    generation_done = False
    max_steps = 100
    step_count = 0
    agent_count =len(decision_steps.agent_id)
    agents = list(range(agent_count))
    removed_agents = []
    while not generation_done:
        # Input my data and get result from network
        actions = np.zeros(shape=(agent_count, 2))
        # print('agents req decision', len(decision_steps))
        if len(decision_steps) > 0:
            for agent_index in agents:
                if agent_index in decision_steps and agent_index < len(nets):
                    action = nets[agent_index].activate(decision_steps[agent_index].obs[0])
                    actions[agent_index] = action

            # print('actions', actions)
            # print( 'action type', type(actions))
            # print('observation 0', decision_steps.obs[0][0])
            # print('observation 1', decision_steps.obs[0][1])
            # print(decision_steps.agent_id)

            # Set the actions
            if len(decision_steps.agent_id) != 0:
                env.set_actions(behavior_name, actions)

        # Move the simulation forward
        env.step()
        # Get the new simulation result
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agents_to_remove = []
        for agent_index in agents:
            if agent_index in decision_steps and agent_index < len(genomes):
                reward = decision_steps[agent_index].reward
                genomes[agent_index][1].fitness += reward.item()
                # print(f"agent {agent_index} fitness: {reward}")
            if agent_index in terminal_steps and agent_index < len(genomes):
                reward = terminal_steps[agent_index].reward
                genomes[agent_index][1].fitness += reward.item()
                # print(f"agent {agent_index} fitness: {reward}")
                agents_to_remove.append(agent_index)

        for agent_index in agents_to_remove:
            agents.remove(agent_index)
            removed_agents.append(agent_index)

        step_count += 1
        # print('step count', step_count)
        if len(removed_agents) >= agent_count or step_count > max_steps:  # all agents terminated
            for agent_index in agents:
                if agent_index in decision_steps and agent_index < len(genomes):
                    genomes[agent_index][1].fitness = genomes[agent_index][1].fitness / step_count
                if agent_index in terminal_steps and agent_index < len(genomes):
                    genomes[agent_index][1].fitness = genomes[agent_index][1].fitness / step_count

            for agent_index in agents_to_remove:
                if agent_index in decision_steps and agent_index < len(genomes):
                    genomes[agent_index][1].fitness = genomes[agent_index][1].fitness / step_count
                if agent_index in terminal_steps and agent_index < len(genomes):
                    genomes[agent_index][1].fitness = genomes[agent_index][1].fitness / step_count

            generation_done = True
            print('agents ended:', len(removed_agents))
            env.reset()





if __name__ == "__main__":
    # Set configuration file
    config_path = "./config"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    best_genome = p.run(run_agent, 2000)

    # Save best genome.
    with open('result/best_genome', 'wb') as f:
        pickle.dump(best_genome, f)

    print(best_genome)

    visualize.plot_stats(stats, view=True, filename="result/feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="result/feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, best_genome, True, node_names=node_names)

    visualize.draw_net(config, best_genome, view=True, node_names=node_names,
                       filename="result/best_genome.gv")
    visualize.draw_net(config, best_genome, view=True, node_names=node_names,
                       filename="result/best_genome-enabled.gv", show_disabled=False)
    visualize.draw_net(config, best_genome, view=True, node_names=node_names,
                       filename="result/best_genome-enabled-pruned.gv", show_disabled=False, prune_unused=True)