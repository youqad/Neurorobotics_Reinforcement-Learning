---
title: "Tutorial 1: Reinforcement Learning"
author:
- 'Younesse Kaddar'
- 'Kexin Ren'
date: 2018-04-03
tags:
  - lab
  - tutorial
  - exercise
  - reinforcement-learning
  - robotics
  - neuroscience
  - neuro-robotique
  - neurorobotics
  - perrin
abstract: 'Lab: Reinforcement Learning'
---

**Lecturer**: Nicolas Perrin


# 1. Markov Decision Problems


Our problem is a Markov Decision Problems (MDP) which is described by a tuple $(𝒳, 𝒰, 𝒫_0, 𝒫,r,γ)$ where $𝒳$ is the state-space, $𝒰$ the action-space, $𝒫_0$ the distribution of the initial state, $𝒫$ the transition function, $r$ the reward function, and $γ ∈ [0, 1]$ a parameter called the discount factor.

The state in which the robot starts, denoted $x_0$, is drawn from the initial state distribution $𝒫_0$. We will choose it such that the robot always starts in state $0$. Then, given the state $x_t$ at time $t$ and the action $u_t$ at that same time, the next state $x_{t+1}$ depends only on $x_t$ and $u_t$ according to the transition function $𝒫$:

$$𝒫(x_t, u_t, x_{t+1}) = {\rm P}(x_{t+1} \mid x_t, u_t)$$

If the robot tries to get out of the grid (for example if it uses action $N$ in state $0$), it will stay in his current state. The robot also receives a reward $r_t$ according to the reward function $r(x_t, u_t)$, which depends on the state $x_t$ and the action $u_t$. We choose to give a reward $1$ when the robot is in state $15$ and does the action $NoOp$, and $0$ for any other state-action pair.


## Questions


### 1. Define `self.gamma`

We may set $γ$ to be equal to $0.95 ∈ [0, 1]$, for instance.

```python
class mdp():
    def __init__(self):
        # [...]
        self.gamma = 0.95
```

### 2. In `__init__(self)`, define `P0`, the distribution of the initial state.

As the robot always starts in state $0$: $𝒫_0$ is equal to the Dirac measure $δ_0$

```python
class mdp():
    def __init__(self):
        # [...]

        self.P0 = np.zeros(self.nX)
        self.P0[0] = 1

        # [...]
```

### 3. Write down the transition probabilities for each triplet $𝒫(\text{state}, \text{action}, \text{next state})$ for the example above

```python
class mdp():
    def __init__(self):
        # [...]

        self.P = np.empty((self.nX,self.nU,self.nX))
        self.P[:,N,:]=  np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        self.P[:,S,:] =  self.P[::-1,N,::-1]

        self.P[:,E,:] = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        self.P[:,W,:] = self.P[::-1,E,::-1]

        self.P[:,NoOp,:]=  np.eye(self.nX)

        # [...]
```


### 4. Define the reward function r, stored in the matrix `self.r`

As reward is equal to $1$ if the robot is in state $15$ and does the action $NoOp$, and $0$ otherwise:

```python
class mdp():
    def __init__(self):
        # [...]
        self.r = np.zeros((self.nX, self.nU))
        self.r[15, NoOp] = 1
```

# 2. Dynamic Programming

The overall goal is to maximize

$$𝔼\left[ \sum\limits_{ t=0 }^∞ γ^t r(x_t, u_t) \mid x_0 \sim 𝒫_0, \; x_{t+1} \sim 𝒫(x_t, u_t, \bullet)  \right]$$

A policy is a mapping $π : X ⟶ U$ that assigns, to each state $x$, an action $u ≝ π(x)$ that the robot should execute whenever in state $x$.

The state value of a policy $π$ is the total discounted reward that the robot expects to receive when starting from a given state $x$ and following policy $π$:

$$V^π(x) ≝ 𝔼\left[ \sum\limits_{ t=0 }^∞ γ^t r(x_t, u_t) \mid x_0 = x, \; u_t = π(x_t), \; x_{t+1} \sim 𝒫(x_t, u_t, \bullet)  \right]$$

The state-action value of a policy $π$ is the total discounted reward that the robot expects to receive when starting from a given state $x$, taking the action $u$ and then following policy $π$:


$$Q^π(x, u) ≝ 𝔼\left[ \sum\limits_{ t=0 }^∞ γ^t r(x_t, u_t) \mid x_0 = x, \; u_0 = u, \, u_t = π(x_t), \; x_{t+1} \sim 𝒫(x_t, u_t, \bullet)  \right]$$

The optimal policy is the policy $π^\ast$ such that for every state:

$$∀x ∈ 𝒳, ∀ π ∈ 𝒰^𝒳, \; V^\ast(x) ≥ V^π(x)$$

In particular, we have:


$$\begin{cases}
  V^\ast(x) = \max_u Q^\ast(x, u)  \\
  π^\ast(x) = {\rm argmax}_u Q^\ast(x, u)  \qquad (3)\\
\end{cases}$$


## 2.1 Value Iteration

It stems from the above that:

$$Q^\ast = r(x, u) + γ \sum\limits_{ y ∈ 𝒳 } 𝒫(x, u, y) \underbrace{V^\ast(y)}_{= \, \max_u Q^\ast(x, u)}$$

out of which we get a recursive algorithm to compute $Q^\ast$:

$$Q^{(k+1)} = r(x, u) + γ \sum\limits_{ y ∈ 𝒳 } 𝒫(x, u, y) \, \max_u Q^{(k)}(x, u) \qquad (8)$$

### 1. Set the proper ranges in the `for i` and `for j` loops.

- `i` ranges over the states
- `j` ranges over the actions

therefore:

```python
def VI(self):
    Q = np.zeros((self.nX,self.nU))
    # [...]

        for i in range(self.nX):
            for j in range(self.nU):
                # [...]
```

### 2. Complete the assignment `Q[i,j] = `


Based on the equation $(8)$:

```python
Q[i,j] = self.r[i,j] + self.gamma * np.sum(self.P[i,j,:] * Qmax)
```

As a result, the `VI` method becomes:

```python
def VI(self):
    Q = np.zeros((self.nX, self.nU))
    pol = N*np.ones((self.nX, 1))
    quitt = False
    iterr = 0

    while quitt==False:
        iterr += 1
        Qold = Q.copy()
        Qmax = Qold.max(axis=1)

        for i in range(self.nX):
            for j in range(self.nU):
                Q[i,j] = self.r[i,j] + self.gamma * np.sum(self.P[i,j,:] * Qmax)

        Qmax = Q.max(axis=1)
        pol =  np.argmax(Q,axis=1)
        if (np.linalg.norm(Q-Qold))==0:
            quitt = True
    return [Q,pol]
```

and by running `python run.py`, the following figure is displayed:

![VI](https://i.gyazo.com/92851318f58af7b738c8b2c875ae38f7.png)


### 3. *Add a second goal:* modify the reward function to give $0.9$ when the robot is in state $5$ and does not move. Explain when and why the robot chooses to go to this new goal instead of the first one. Explain what parameter is responsible of this behaviour.

The choice of the discount factor $γ$ is a tradeoff between exploration (exploring farther states) and exploitation/greediness (exploiting the rewards of the nearby one). As it happens: the smaller the parameter $γ$, the more the robot tends to exploit the closest state associated with a (strictly) positive reward (even if there might be a state farther on which a given action leads to a bigger reward).

Here, there are two different optimal policies, depending on the value of $γ$:


|Value of $γ$|Optimal Policy|
-|-
**Greedier policy**: $γ < γ_0 ≝ \sqrt{0.9}$|![](https://i.gyazo.com/58a8c0b73b861fef7e78968862a88a01.png)
**More exploratory policy**: $γ ≥ γ_0$|![](https://i.gyazo.com/c2672fedffe41477ec1e465379986ecb.png)


When it comes to the greedier policy: for states close to the state $5$ (denoted by $6$ on the pictures), the robot tends to head to the state $5$, even if its reward ($=0.9$) is inferior the reward ($=1$) of the state $15$ (denoted by $16$ on the pictures)

On the contrary, with the more exploratory policy: apart from the states $0, 1$ and $4$ (which are one step away from the state $5$), the robot favors the state $15$, i.e. the long-term bigger reward over the smaller yet *closer* (*for the states $2, 6, 8$ and $9$*) reward of the state $5$.

### 4. Change `self.P` to implement stochastic (non-deterministic) transitions. Use comments in the code to describe the transitions you chose and the results you observed while running `VI` on the new transitions you defined.
