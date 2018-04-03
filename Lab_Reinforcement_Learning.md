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

$$Q^\ast = r(x, u) + γ \sum\limits_{ y ∈ 𝒳 } 𝒫(x, u, y) \underbrace{V^\ast(y)}_{= \, \max_u Q^\ast(x, u)} \qquad ⊛$$

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
**Greedier policy**: $γ < γ_0 ≝ \sqrt{0.9}$|![Gamma_smaller_than_square_root_0.9 ](https://i.gyazo.com/58a8c0b73b861fef7e78968862a88a01.png)
**More exploratory policy**: $γ ≥ γ_0$|![Gamma_greater_than_square_root_0.9 ](https://i.gyazo.com/c2672fedffe41477ec1e465379986ecb.png)


When it comes to the greedier policy: for states close to the state $5$ (denoted by $6$ on the pictures), the robot tends to head to the state $5$, even if its reward ($=0.9$) is inferior the reward ($=1$) of the state $15$ (denoted by $16$ on the pictures)

On the contrary, with the more exploratory policy: apart from the states $0, 1$ and $4$ (which are one step away from the state $5$), the robot favors the state $15$, i.e. the long-term bigger reward over the smaller yet *closer* (*for the states $2, 6, 8$ and $9$*) reward of the state $5$.

### 4. Change `self.P` to implement stochastic (non-deterministic) transitions. Use comments in the code to describe the transitions you chose and the results you observed while running `VI` on the new transitions you defined.
To implement stochastic transition, we use the first state and the seventh state as expamples by the following codes (tried to make every state stochastic but that goes too complecated when one stochastic state is enough to expound this question):

```python
  pos = np.random.rand(5)	
  n = 0 #n = 0 for 1st state, n = 6 for 7th state
  for i in range(self.nU):
		self.P[n,i,np.where(self.P[n,i,:]==1)]=pos[i]/sum(pos)
 ```
We found that, when the first step (at state 0) is stochastic, the result is the same as the deterministic one (as shown in Figure 2.1.4.1) which makes sense because all routes will actually weight the same under these conditions; when the 7th state (state 6) is stochastic, no neighbor state of state 6 will choose to go through state 6 (as shown in Figure 2.1.4.2), becuase there are posibilities of going "backwards" (i.e. not toward to the final state, state 15) on state 6 which will reduce the Q of other states moving to state 6.

![Figure 2.1.4.1](https://github.com/youqad/Neurorobotics_Reinforcement-Learning/blob/master/2.png?raw=true "Figure 2.1.4.1")
![Figure 2.1.4.2](https://github.com/youqad/Neurorobotics_Reinforcement-Learning/blob/master/1.png?raw=true "Figure 2.1.4.2")


## 2.2. Policy Iteration


By definition of the state value function of a given policy $π$, we have:

$$V^π(x) = r(x, π(x)) + γ \sum\limits_{ y ∈ 𝒳 } 𝒫(x, π(x), y) V^π(y) \qquad ⊛⊛$$

But as $𝒳$ is finite: by setting <span>$\textbf{V}_π$ (resp. $\textbf{R}_π$) to be the vector-matrix $(V^π(x))_{x ∈ 𝒳}$ (resp. $(r(x, π(x)))_{x ∈ 𝒳}$)</span>, and

$$\textbf{P}_π ≝ (𝒫(x, π(x), y))_{\substack{x ∈ 𝒳 \\ y ∈ 𝒳}}$$

it comes that

$$\begin{align*}
& \; \textbf{V}_π = \textbf{R}_π + γ \textbf{P}_π \textbf{V}_π\\
⟺ & \; \textbf{V}_π =(\textbf{I} - γ \textbf{P}_π)^{-1} \textbf{R}_π  && (9)\\
\end{align*}
$$

which yields another algorithm to compute the optimal policy, along with

$$\begin{cases}
Q^π(x, u) = r(x, u) + γ \sum\limits_{ y ∈ 𝒳 } 𝒫(x, u, y) V^π(y) && (6)\\
π^{(k+1)}(x) = {\rm argmax}_u {Q^π}^{(k)}(x, u)  && (10)\\
\end{cases}$$

### 1. Fill the code `PI(self)` to implement policy iteration, and test it. Compare the converge speed of `VI` and `PI`.

Using the previous equations, it comes that:

```python
def PI(self):
    Q = np.empty((self.nX, self.nU))
    pol = N*np.ones(self.nX, dtype=np.int16)
    I = np.eye((self.nX))
    R = np.zeros(self.nX)
    P = np.zeros((self.nX, self.nX))
    quitt = False
    iterr = 0

    while quitt==False:
        iterr += 1

        for i in range(self.nX):
            R[i] = self.r[i,pol[i]]
            for j in range(self.nX):
                P[i,j] = self.P[i,pol[i],j]

        V = np.dot(np.linalg.inv(I-self.gamma*P), R)

        for i in range(self.nX):
            for j in range(self.nU):
                Q[i,j] = self.r[i,j] + self.gamma * np.sum(self.P[i,j,:] * V)
                print(Q[i,j])

        pol_old = pol.copy()
        pol = np.argmax(Q, axis=1)

        if np.array_equal(pol, pol_old):
            quitt = True

    return [Q, pol]
```

We compared the convergences of VI and PI and found that, VI algorithm converges after 667 iterations while the PI algorithm converges after 3 iterations, supporting that PI is more efficient than VI.

# 3. Reinforcement Learning: Model-free approaches

## 3.1. Temporal Difference Learning (TD-learning)

Based on $⊛$ and $⊛⊛$, we get:

$$\begin{cases}
V^π(x) = 𝔼\left[r(x, π(x)) + γ V^π(y) \mid y \sim 𝒫(x, π(x), \bullet)\right] &&(11)\\
Q^\ast(x, u) = 𝔼\left[r(x, u) + γ \max_v Q^\ast(y, v) \mid y \sim 𝒫(x, u, \bullet)\right] &&(12)\\
\end{cases}$$

as a result of which we define the *temporal difference errors* (TD errors):

$$\begin{cases}
δ_{V^π}(x) ≝ 𝔼\left[r(x, π(x)) + γ V^π(y) - V^π(x) \mid y \sim 𝒫(x, π(x), \bullet)\right] &&(13)\\
δ_{Q^\ast}(x, u) ≝ 𝔼\left[r(x, u) + γ \max_v Q^\ast(y, v) - Q^\ast(x, u) \mid y \sim 𝒫(x, u, \bullet)\right] &&(14)\\
\end{cases}$$


## 3.2. TD(0)

At time $t$, we denote by

- $V^{(t})$ the estimation of the state value function
- $x_t$ the current state
- <span>$δ_t ≝ δ_{V^{(t)}}(x_t)$</span>
- $α$ the learning rate

From the $δ$-rule, the following update of the state value function can be derived:

$$V^{(t+1)}(x_t) = V^{(t)}(x_t) + α \underbrace{δ_t}_{\rlap{≝ \; r(x_t, u_t) + γ V^{(t)}(x_{t+1}) - V^{(t)}(x_t)}} \quad\qquad (15)$$


### 1. Open the file `mdp.py`, and implement a function called `MDPStep`. This function returns a next state and a reward given a state and an action. *Hint*: To draw a number according to a vector of probabilities, you can use the function `discreteProb`.

```python
def MDPStep(self,x,u):
    # This function executes a step on the MDP M given current state x and action u.
    # It returns a next state y and a reward r
    y = self.discreteProb(self.P[x,u,:]) # y is sampled according to the distribution self.P[x,u,:]
    r = self.r[x,u] # r is be the reward of the transition
    return [y,r]
```


### 2. Fill-in the missing lines in the `TD` function, which implements $TD(0)$. To obtain samples from MDP, you will use the `MDPStep` function

```python
def TD(self,pol):
    V = np.zeros((self.nX,1))
    nbIter = 100000
    alpha = 0.1
    for i in range(nbIter):
        x = np.floor(self.nX*np.random.random())
        [y, r] = self.MDPStep(x, pol[x])
        V[x] += alpha * (r + self.gamma * V[y] - V[x])
    return V
```

By

- modifying `run.py` so that one computes:

    ```python
    [Q,pol] = m.PI()
    V = m.TD(pol)
    ```

- and adding the line

    ```python
    print(np.argsort(V,axis=0))
    ```
    in the `TD` method, just before the return, to sort the states according to their estimated value (in increasing order)

it appears that the estimated values are sorted as follows:

$$
\begin{align*}
    \quad & V^π(1) \;\, ≤ V^π(3) \;\, ≤ V^π(9) \;\, ≤ \;\,  V^π(4)\\
    ≤ \; & V^π(13) ≤ V^π(2) \;\, ≤ V^π(5) \;\, ≤ V^π(7)\\
    ≤ \; & V^π(10) ≤ V^π(8) \;\, ≤ V^π(11) ≤ V^π(14)\\
    ≤ \; & V^π(6) \;\, ≤ V^π(12) ≤ V^π(15) ≤ V^π(16)
\end{align*}
$$

for

- the deterministic transition function and the reward function of question **2.1.3.** (the reward is null everywhere except at state $5$ (where it amounts to $0.9$) and state $15$ (where it amounts to $1$) when the robot doesn't move)

- the following policy ($γ < \sqrt{0.9}$): ![Gamma_smaller_than_square_root_0.9 ](https://i.gyazo.com/58a8c0b73b861fef7e78968862a88a01.png)

which makes perfect sense, intuitively.



### 3. Write a function `compare` that takes in input a state value function $V$, a policy $π$, a state-action value function $Q$, and returns `True` if and only if $V$ and $Q$ are consistent with respect to $π$ up to some precision, i.e. if $∀x ∈ 𝒳, V^π(x) = Q^π(x, π(x)) ± ε$.

We are asked to program a function which test if $∀x ∈ 𝒳, \vert V^π(x) - Q^π(x, π(x)) \vert ≤ ε_{max}$, for a threshold value $ε_{max}$: i.e. if the infinity norm of the difference of the vectors is smaller than $ε_{max}$.

```python
def compare(self,V,Q,pol,eps=0.0001):
    Q_pol = np.array([[Q[x, pol[x]]] for x in range(V.size)])
    return np.linalg.norm(V-Q_pol, ord=np.inf)<eps
```

### 4. Use the `compare` function to verify that $TD(0)$ converges towards the proper value function, using it on the policy returned by `VI` or `PI`.

```python
[Q1,pol1] = m.VI()
[Q2,pol2] = m.PI()

V1 = m.TD(pol1)
V2 = m.TD(pol2)

print(m.compare(V1,Q1,pol1))
print(m.compare(V2,Q2,pol2))
```

yields `True` and `True`, so $TD(0)$ does converge towards the proper value function.

## 3.3. Q-Learning

In accordance with the softmax-policy:

$$π^{(t)}(u \mid x) ≝ \frac{\exp(Q^{(t)}(x, u)/τ)}{\sum\limits_{ v ∈ 𝒰 } \exp(Q^{(t)}(x, v)/τ)}$$

we update the state-action value function as follows:

$$Q^{(t+1)}(x_t, u_t) = Q^{(t)}(x_t, u_t) + α \left[r(x_t, u_t) + γ \max_{u_{t+1} ∈ 𝒰} Q^{(t)}(x_{t+1}, u_{t+1}) - Q^{(t)}(x_t, u_t)\right] \qquad (16)$$

### 1. Implement the function `softmax` that returns the soft-max policy.

```python
def softmax(self,Q,x,tau):
    # Returns a soft-max probability distribution over actions
    # Inputs :
    # - Q : a Q-function reprensented as a nX times nU matrix
    # - x : the state for which we want the soft-max distribution
    # - tau : temperature parameter of the soft-max distribution
    # Output :
    # - p : probabilty of each action according to the soft-max distribution
    # (column vector of length nU)

    p = np.exp(Q[x,:]/tau)
    return p/np.sum(p)
```

### 2. Fill-in the missing lines to implement the `QLearning` function.

```python
def QLearning(self,tau):
    # This function computes the optimal state-value function and the corresponding policy using Q-Learning

    # Initialize the state-action value function
    Q = np.zeros((self.nX,self.nU))

    # Run learning cycle
    nbIter = 100000
    alpha = 0.01
    for i in range(nbIter):
        # Draw a random state
        x = np.floor(self.nX*np.random.random())

        # Draw an action using a soft-max policy
        u = self.discreteProb(self.softmax(Q,x,tau))

        # Perform a step of the MDP
        [y,r] = self.MDPStep(x, u)

        # Update the state-action value function with Q-Learning
        Q[x,u] += alpha * (r + self.gamma * np.max(Q[y,:]) - Q[x,u])

    # Compute the corresponding policy
    Qmax = Q.max(axis=1)
    pol =  np.argmax(Q,axis=1)
    return [Qmax,pol]
```

### 3. Run Q-Learning several times. What do you observe?

TODO

### 4. Compare the state-value function and the policy computed using Q-Learning with the ones you obtained with VI and PI.

TODO


# 4. Model-Based Reinforcement Learning (MBRL)

**Real-Time Dynamic Programming**:

We update the estimate of the transition probabilty $\hat{𝒫}^{(t)}$ as follows:

$$\hat{𝒫}^{(t)}(x_t, u_t, y) ≝ \left(1 - \frac 1 {N_t(x_t, u_t)}\right) \hat{𝒫}^{(t)}(x_t, u_t, y) + \frac 1 {N_t(x_t, u_t)} \mathbb{1}_{y = x_{t+1}} \qquad (17)$$

where $N_t(x,u)$ is the number of visits to the pair $(x,u)$ before (and including) time $t$.

As for the reward function estimate:

$$\hat{r}^{(t+1)}(x_t, u_t) ≝ r_t \qquad (18)$$


Those are used in the following updating rule:

$$Q^{(t+1)}(x_t, u_t) ≝ \hat{r}^{(t+1)}(x_t, u_t) + γ \sum\limits_{ y ∈ 𝒳 } \hat{𝒫}^{(t+1)}(x_t, u_t, y) \max_{v ∈ 𝒰} Q^{(t)}(y, v) \qquad (19)$$

### 1. Fill-in `RTDP` (Real-Time Dynamic Programming) function

```python
def RTDP(self):
    Q = np.zeros((self.nX,self.nU))
    hatP = np.ones((self.nX,self.nU,self.nX))/self.nX
    N = np.ones((self.nX,self.nU))

    I = np.array(range(self.nX))

    nbIter = 10000

    for iterr in range(nbIter):
        # Draw a random pair of state and action
        x = np.floor(self.nX*np.random.random())
        u = np.floor(self.nU*np.random.random())

        # One step of the MDP for this state-action pair
        [y,r] = self.MDPStep(x,u)

        # Compute the estimate of the transition probabilities
        hatP[x,u,:] *= (1 - 1/N[x, u])
        hatP[x,u,:] += (np.arange(self.nX) == y).astype(int)/N[x, u]

        # Updating rule for the state-action value function
        Qmax = Q.max(axis=1)
        Q[x,u] = r + self.gamma * np.sum(hatP[x,u,:]*Qmax)

        N[x,u] += 1

    Qmax = Q.max(axis=1)
    pol =  np.argmax(Q,axis=1)

    return [Qmax,pol]
```
