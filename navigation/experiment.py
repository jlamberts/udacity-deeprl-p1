import pathlib
import time
import inspect
import json
import shutil
from collections import deque
import torch
import numpy as np
from navigation.agent import Agent


def dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        env (UnityEnvironment): pre-initialized UnityEnvironment
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    brain_name = env.brain_names[0]
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            # perform step
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)))
    return scores


def run_and_save_experiment(experiment_name, initialized_env, experiment_kwargs=None, agent_kwargs=None, experiment_dir="experiments"):
    # Default to no kwargs; don't want to actually set a mutable as a default function arg
    experiment_kwargs = experiment_kwargs if experiment_kwargs else {}
    agent_kwargs = agent_kwargs if agent_kwargs else {}

    # check whether the experiment dir exists and is populated; verify with user before deleting if so
    target_dir = pathlib.Path(experiment_dir) / experiment_name
    if target_dir.is_dir() and len(list(target_dir.iterdir())) > 0:
        print(f"Directory '{target_dir}' already exists and is not empty.  Overwrite it?")
        user_response = input("Type 'Yes' to overwrite, or 'Skip' to skip this experiment.\n")
        if user_response.lower() in {"'skip'", "skip"}:
            return None
        elif user_response.lower() not in {"'yes'", "yes"}:
            raise RuntimeError("User declined to overwrite the data in the directory.")
        else:
            shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    experiment_json_path = target_dir / "experiment.json"
    experiment_model_weights_path = target_dir / "model_weights.pt"

    # Get environment metadata
    brain_name = initialized_env.brain_names[0]
    brain = initialized_env.brains[brain_name]

    # reset the environment
    env_info = initialized_env.reset(train_mode=True)[brain_name]

    # Initialize agent
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    # initialize agent
    agent = Agent(state_size=state_size, action_size=action_size, **agent_kwargs)

    # train_agent
    start_time = time.time()
    historical_scores = dqn(agent, env=initialized_env, **experiment_kwargs)
    run_time_seconds = time.time() - start_time
    print(f"\ntraining finished for experiment {experiment_name} in {run_time_seconds} seconds")

    # save model weights
    torch.save(agent.qnetwork_local.state_dict(), experiment_model_weights_path)

    # combine default experiment kwargs with user-provided overrides
    experiment_params = {k: v.default for k, v in inspect.signature(dqn).parameters.items() if v.default is not inspect.Parameter.empty}
    experiment_params.update(experiment_kwargs)

    # combine default agent kwargs with user-provided overrides
    agent_params = {
        k: v.default for k, v in inspect.signature(Agent).parameters.items() if v.default is not inspect.Parameter.empty and k != "env"
    }
    agent_params.update(agent_kwargs)
    agent_params["network"] = agent_params["network"].__name__

    # serialize experiment metadata and results (this should probably be a class)
    experiment_dict = {
        "experiment_name": experiment_name,
        "results": {"run_time_seconds": run_time_seconds, "scores": historical_scores},
        "params": {"experiment": experiment_params, "agent": agent_params},
    }

    with open(experiment_json_path, "w") as f:
        json.dump(experiment_dict, f)


def load_network_from_experiment(experiment_path):
    experiment_dir_path = pathlib.Path(experiment_path)
    experiment_json_path = experiment_dir_path / "experiment.json"
    experiment_weights_path = experiment_dir_path / "model_weights.pt"

    with open(experiment_json_path, "r") as f:
        experiment_dict = json.load(f)

    network_name_map = {"QNetwork": QNetwork, "DuelingQNetwork": DuelingQNetwork}
    network_class = network_name_map[experiment_dict["params"]["agent"]["network"]]

    if experiment_dict["params"]["agent"]["network_kwargs"]:
        network_kwargs = experiment_dict["params"]["agent"]["network_kwargs"]
    else:
        network_kwargs = {}

    network = network_class(**network_kwargs)
    network.load_state_dict(torch.load(experiment_weights_path))
    return network


def load_scores_from_experiment(experiment_path):
    experiment_dir_path = pathlib.Path(experiment_path)
    experiment_json_path = experiment_dir_path / "experiment.json"

    with open(experiment_json_path, "r") as f:
        experiment_dict = json.load(f)

    return experiment_dict["results"]["scores"]


def load_runtime_from_experiment(experiment_path):
    experiment_dir_path = pathlib.Path(experiment_path)
    experiment_json_path = experiment_dir_path / "experiment.json"

    with open(experiment_json_path, "r") as f:
        experiment_dict = json.load(f)

    return experiment_dict["results"]["run_time_seconds"]
