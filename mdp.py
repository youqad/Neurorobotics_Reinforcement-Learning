import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

##########-ACTIONS DEFINITION-###################
N = 0
S = 1
E = 2
W = 3
NoOp = 4

class mdp():

######################-MDP DEFINITION-###########

    def __init__(self):
        self.nX = 16
        self.nU = 5

        self.P0 = np.zeros(self.nX)
        self.P0[0] = 1

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

        self.r = np.zeros((self.nX, self.nU))
        self.r[15, NoOp] = 1
        self.r[5, NoOp] = 0.9


        self.gamma = np.sqrt(0.9)-0.000000000000001
        #self.gamma = "?"

###################-DYNAMIC PROGRAMMING-###############


    def VI(self):
        Q = np.zeros((self.nX, self.nU))
        pol = N*np.ones(self.nX)
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


##############-TD-LEARNING-########################
    def TD(self,pol):
        V = np.zeros((self.nX,1))
        nbIter = 100000
        alpha = 0.1
        for i in range(nbIter):
            x = np.floor(self.nX*np.random.random()).astype(int)
            [y, r] = self.MDPStep(x, pol[x])
            V[x] += alpha * (r + self.gamma * V[y] - V[x])
        print(np.argsort(V,axis=0))
        return V

    def discreteProb(self,p):
        # Draw a random number using probability table p (column vector)
        # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1 and the components p(j) are nonnegative.
        # To generate a random sample of size m from this distribution imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n).
        # Generate a uniform number rand, if this number falls in the jth interval give the discrete distribution the value j. Repeat m times.
        r = np.random.random()
        cumprob=np.hstack((np.zeros(1),p.cumsum()))
        sample = -1
        for j in range(p.size):
            if (r>cumprob[j]) & (r<=cumprob[j+1]):
                sample = j
                break
        return sample

    def MDPStep(self,x,u):
	    # This function executes a step on the MDP M given current state x and action u.
        # It returns a next state y and a reward r
        y = self.discreteProb(self.P[x,u,:]) # y is sampled according to the distribution self.P[x,u,:]
        r = self.r[x,u] # r is be the reward of the transition
        return [y,r]

    def compare(self,V,Q,pol):
       sumval = 0
       #for i in range(V.size):
       #    sumval += abs(V[i] - Q[i,pol[i]])
       return sumval

##################-Q-LEARNING-############

    def softmax(self,Q,x,tau):
        # Returns a soft-max probability distribution over actions
        # Inputs :
        # - Q : a Q-function reprensented as a nX times nU matrix
        # - x : the state for which we want the soft-max distribution
        # - tau : temperature parameter of the soft-max distribution
        # Output :
        # - p : probabilty of each action according to the soft-max distribution
        #(column vector of length nU)

        p = np.zeros((self.nU))
        '...'
        return p

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
            #[y,r] = "?"

            # Update the state-action value function with Q-Learning
            #Q[x,u] = "?"

        # Compute the corresponding policy
        Qmax = Q.max(axis=1)
        pol =  np.argmax(Q,axis=1)
        return [Qmax,pol]

##################-RTDP-#####################

    def RTDP(self):
        Q = np.zeros((self.nX,self.nU))
        hatP = np.ones((self.nX,self.nU,self.nX))/self.nX
        N = np.ones((self.nX,self.nU))

        I = np.array(range(self.nX))

        nbIter = 10000

        for iterr in range(nbIter):
            x = "?"
            #u = np.floor(self.nU*np.random.random())
            #[y,r] = self.MDPStep(x,u)
            #hatP[x,u,:] = "?"
            #Qmax = Q.max(axis=1)
            #Q[x,u] = "?"
            #N[x,u] = N[x,u]+1

        Qmax =Q.max(axis=1)
        pol =  np.argmax(Q,axis=1)

        return [Qmax,pol]


    def RTDP2(self):

        Q = np.zeros((self.nX,self.nU))
        hatP = np.ones((self.nX,self.nU,self.nX))/self.nX
        hatR = np.zeros((self.nX,self.nU))
        N = np.ones((self.nX,self.nU))

        I = np.array(range(self.nX))

        nbIter = 10000

        for iterr in range(nbIter):
            x = "?"
            #u = np.floor(self.nU*np.random.random())
            #[y,r] = self.MDPStep(x,u)
            #hatP[x,u,:] = "?"
            #hatR[x,u]= "?"
            #Qmax = Q.max(axis=1)
            #Q[x,u] = "?"
            #N[x,u] = N[x,u]+1

        Qmax =Q.max(axis=1)
        pol =  np.argmax(Q,axis=1)

        return [Qmax,pol]

###################-PLOT-#################

def statePos(i):
	return (0.05 + (((i-1) // 4)/10.0), 0.35 - ((i-1) % 4)/10.0)

def statePos_text(i):
	return (statePos(i)[0] - 0.025, statePos(i)[1] + 0.01)
def arrow(i,j):
	dx = 0.9 *(statePos(j)[0] - statePos(i)[0])
	dy = 0.9 * (statePos(j)[1] - statePos(i)[1])
	x0 = statePos(i)[0] + 1.7*dx/18.0
	y0 = statePos(i)[1] + 1.7*dy/18.0
	return (x0, y0, dx, dy)

def plotPolicy(m,pol):
	fig, ax = plt.subplots()
	plt.vlines(0,0.0,0.4,lw=3.5)
	plt.vlines(0.4,0.0,0.4,lw=3.5)
	plt.hlines(0,0.0,0.4,lw=3.5)
	plt.hlines(0.4,0.0,0.4,lw=3.5)
	for i in range(1,17):
		print(m.P[i-1,pol[i-1,0],:])
		arr = arrow(i,1+m.P[i-1,pol[i-1,0],:].argmax())
		if( arr[2] != 0.0 or arr[3] != 0.0):
			plt.arrow(*arr, length_includes_head = True, width=0.005)
		else:
			circle = mpatches.Circle(statePos(i), 0.01, ec="none")
			ax.add_patch(circle)
	for i in range(1,17):
		plt.annotate(i, xy=(0,0), xytext=statePos_text(i))

	plt.xticks([i/10.0 for i in range(0,5)], visible = False )
	plt.yticks([i/10.0 for i in range(0,5)], visible = False )
	plt.grid(which='both', ls='--')
	plt.axis('equal')
	plt.axis([0.0,0.4,0.0,0.4])
	plt.show(fig)
