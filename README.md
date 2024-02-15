
# Project Title
Deep Q-Network for Demon Attack

## Table of Contents
Introduction
Project Description
Data Analysis
Technologies Used
Installation
Usage
Contributing

## Introduction


This project implements a Deep Q-Network (DQN) agent for playing the Atari game Demon Attack using reinforcement learning techniques. The DQN agent learns to play the game by interacting with the game environment and receiving rewards based on its actions. The goal of the agent is to maximize its cumulative reward over time by learning an optimal policy for making decisions in the game.

### Policy and SARSA for AI Explanation:

In reinforcement learning, a policy is a strategy that an agent follows to make decisions in an environment. It maps states of the environment to actions that the agent can take. The policy determines the behavior of the agent and influences how it interacts with the environment.

SARSA (State-Action-Reward-State-Action) is a reinforcement learning algorithm similar to Q-learning, but instead of learning the value of state-action pairs (Q-values), it learns the value of state-action-next state triples. This allows SARSA to learn directly from experience and update its policy while exploring the environment.

In the context of this project, the DQN agent learns an optimal policy for playing the Demon Attack game by using a deep neural network to approximate the Q-values of state-action pairs. By iteratively interacting with the game environment, the agent learns to update its policy based on the observed rewards and transitions between states. This allows the agent to gradually improve its performance and achieve higher scores in the game.


## Project Description
This project focuses on developing a Deep Q-Network (DQN) agent to play the Atari game Demon Attack using reinforcement learning techniques. The main objective is to train an artificial intelligence (AI) agent capable of learning to play the game effectively by interacting with the game environment and maximizing its cumulative reward over time.

### Goals and Objectives:

1. Develop a DQN agent capable of learning to play the Demon Attack game.
2. Train the agent using reinforcement learning techniques to improve its performance over time.
3. Implement a robust methodology for training and evaluating the agent's performance.
4. Optimize the agent's learning process to achieve high scores in the game.

## Data Analysis
In the data analysis section, the primary focus is on collecting, understanding, and structuring the information gathered from the Atari game environment, particularly in the context of the Demon Attack game. The provided code snippets showcase the examination of the action space and observation space of the game environment, which are crucial for developing and training the AI agent.

Action Space: The action space represents the possible actions that the AI agent can take within the game environment. In the case of Demon Attack, the action space is discrete with six possible actions, denoted by integers from 0 to 5.

Observation Space: The observation space defines the format of the observations or states that the AI agent receives from the game environment. In this scenario, the observation space is represented as an n-dimensional box with three numbers, indicating the structure and format of the game state information.

Data Structures: The code further illustrates the organization of collected data, including observations, previous observations, actions, and rewards, into structured lists or arrays. These data structures are essential for storing and managing the information necessary for training and evaluating the AI agent's performance.

Insights: The insights gained from analyzing the data structures reveal the dimensions and characteristics of the collected data, such as the length of observation lists, the shape of action lists, and the type of data stored in these lists (e.g., numpy.uint8 for observation elements).


## Technologies Used
The development and implementation of the Demon Attack AI agent leveraged a variety of technologies and tools tailored to the specific requirements of the project. Below is a breakdown of the key technologies utilized:

Gym: Gym is an open-source toolkit for developing and comparing reinforcement learning algorithms. It provides a wide range of environments, including the Demon Attack environment, allowing for standardized testing and benchmarking of AI agents.

Atari-Py: Atari-Py is a Python library that provides interfaces to the Atari 2600 game emulator. It allows developers to interact with Atari games programmatically, making it suitable for training AI agents using reinforcement learning techniques.

PyTorch: PyTorch is a popular deep learning framework that offers flexible and efficient tools for building and training neural networks. In this project, PyTorch is utilized for implementing the neural network architecture of the AI agent and for defining custom loss functions and optimization strategies.

Matplotlib: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is employed in this project for generating plots and graphs to visualize training progress, performance metrics, and game-related statistics.

NumPy: NumPy is a fundamental package for scientific computing in Python, providing support for multi-dimensional arrays and matrices. It is extensively used for numerical operations and data manipulation, particularly in processing observations and actions within the Demon Attack environment.

TQDM: TQDM is a fast, extensible progress bar library for Python and CLI. It is utilized to create progress bars and monitor the training process, allowing for real-time tracking of iterations and epochs during model training.

IPython: IPython provides an interactive computing environment for Python, offering features such as enhanced introspection, rich media integration, and support for interactive data visualization. It is employed in this project for interactive experimentation and exploration of the codebase.

Pandas: Pandas is a powerful data manipulation and analysis library for Python, offering data structures and operations for manipulating structured data. It is used for organizing and processing data collected during the training and evaluation of the AI agent.

Seaborn: Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. It is employed for creating visually appealing and insightful plots to analyze the performance and behavior of the AI agent.

## Installation
pip install -r requirements.txt

Then you can run the cells in the notebook, the first step is the play the game to collect the RAM data necessary for training. Once you believe you have recieved enough data, you can pass it through the network.
