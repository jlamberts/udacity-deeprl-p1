from unityagents import UnityEnvironment

from navigation.networks import DuelingQNetwork, QNetwork
from navigation.experiment import run_and_save_experiment

# modify this to point to a different env file
UNITY_ENV_FILE = "Banana.app"
# list of tuples containing (experiment_name, agent_parameters); modify to change the experiments run
EXPERIMENT_SETUPS = [
    ("basic_dqn", {"double_dqn": False, "network": QNetwork, "seed": 42}),
    ("dueling_dqn", {"double_dqn": False, "network": DuelingQNetwork, "seed": 42}),
    ("double_dqn", {"double_dqn": True, "network": QNetwork, "seed": 42}),
    ("dueling_double_dqn", {"double_dqn": True, "network": DuelingQNetwork, "seed": 42}),
    (
        "dueling_double_dqn_small_network",
        {"double_dqn": True, "network": DuelingQNetwork, "seed": 42, "network_kwargs": {"hidden_layer_size": 256}},
    ),
]

unity_env = UnityEnvironment(file_name=UNITY_ENV_FILE)
for experiment_name, agent_params in EXPERIMENT_SETUPS:
    print("running experiment:", experiment_name)
    run_and_save_experiment(experiment_name, unity_env, agent_kwargs=agent_params)
