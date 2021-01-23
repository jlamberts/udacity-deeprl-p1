# Udacity Deep Reinforcement Learning Project 1: Navigation

## Project Details
The code in this repo interacts with a modified version of the [Unity Banana Collector Environment.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector)
In this environment, the agent's goal is to collect as many yellow bananas as possible, while avoiding the blue bananas.
For each yellow banana collected, the agent receives a reward of +1, and for each blue banana collected it receives
a reward of -1.

To achieve this goal, the agent has four discrete actions, corresponding to "turn left", "turn right", "move backward", 
and "do nothing".  State information is given to the agent as a vector of length 37; this state information contains
velocity and information about objects that the agent can "see".

The environment is considered "solved" when the agent receives an average score of at least +13 for a window of
100 episodes.

## My Solution
To solve the environment, I implemented three variations on Deep Q Networks: "vanilla" DQN, Double DQN, and Dueling DQN.
I also implemented a basic experiment runner and serialization format to more easily run the different setups and
compare results.  For more details on my findings, see the writeup in [report.ipynb](report.ipynb)

## Getting Started

### Python Setup
This project has been tested on Python 3.6; it may work on later versions but is incompatible with earlier ones.
It is recommended that you use a virtual environment using conda or another tool when installing project dependencies.
You can find the instructions for installing miniconda and creating an environment using conda on the
[conda docs](https://docs.conda.io/en/latest/miniconda.html).

### Python Dependencies
After creating and activating your environment (if you're using one), you should install the dependencies for this project
by following the instructions in the [Udacity DRLND Repository.](https://github.com/udacity/deep-reinforcement-learning#dependencies)


### Unity environment
Once you have the python dependencies installed, download the version of the unity environment appropriate for
your operating system.  Links for each operating system can be found below:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac Os](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

After downloading, use 7zip or another archive tool to extract the environment file into the root project directory.
By default, the code is set up to look for the Mac OS version of the environment, so you will need to modify the
UNITY_ENV_PATH variable in `run_experiments.py` or `run_agent.py` to point to your new version.

## Instructions
The experiment runner, `navigation.experiment.run_and_save_experiment`, accepts dictionaries of keyword args which
modify behavior at the experiment level (number of epochs, epsilon decay rate, etc.) as well as behavior on the agent
level (type of network used, size of hidden layer, double DQN vs "vanilla" Q-value calculations).  By default, the file
`run_experiments.py` is configured to run 5 experiments and write the results to the `experiments` folder.  You can modify
the `EXPERIMENT_SETUPS` list at the top of the file to adjust the experiment setups if you'd like to compare
different setups.  

To run the default experiments, navigate to the root of the project and run the command `python run_experiments.py`.
If there is data in an experiment's folder already, the program will ask you whether you want to overwrite the data or
skip the experiment.  If neither option is chosen, the program will terminate to avoid overwriting the existing
experiment.

The experiments directory contains one folder per experiment run.  Inside each folder are two files: `experiment.json`
and `model_weights.pt`.  `experiment.json` contains experiment metadata, including the parameters used for the experiment,
runtime, and training errors.  `model_weights.pt` contains the final trained weights at the end of the experiment which
can be loaded into a PyTorch model using the command `my_model.load_state_dict(torch.load(weights_path))`.  Alternatively,
you can load the model from the experiment directory without first instantiating a model using the 
`navigation.experiment.load_experiment_model(experiment_folder_path)` function.
