# <p align="center">POLICY EVALUATION</p>

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
<p align="center">
<img width="600" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44"> </p>


## POLICY EVALUATION FUNCTION

### Formula
<img width="350" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e663bd3d-fc85-41c3-9a5c-dffa57eae250">

### Program
```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
   	'''Initialize 1st Iteration estimates of state-value function(V) to zero'''
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        '''Initialize the current iteration estimates to zero'''
        V=np.zeros(len(P),dtype=np.float64)
        
        for s in range(len(P)):
        
            '''Update the value function for each state'''
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                
            '''Check for convergence'''
            if np.max(np.abs(prev_V-V))<theta:
                break
                
            '''Update the previous state-value function'''
            prev_V=V.copy()
        return V
```

## OUTPUT:
### Policy 1
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/0a210271-47e9-4ea5-ab83-35053faaf4a5)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/b3b933fd-60c8-48ec-b45b-07721b2d58bd)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/c8ec140b-ac35-41cb-b068-d2e3f66fbc3b)
### Policy 2
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/c5863b33-0d56-45e6-9836-3b69f9515c4d)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/f1875c9e-91fb-467c-ad63-df77c705ae0a)
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/0435f712-b2a4-44e8-ab64-a37fde842551)
### Comparison
![image](https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/4c85b228-5826-43a1-9733-44a99a57cb42)
### Conclusion
<p align="center">
<img src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/4c0e961c-301e-4d9b-96cc-cdeb490f13c9"> </p>


## RESULT:
Thus, a Python program is developed to evaluate the given policy.
