from mdp import *


m = mdp()


#print '-------------VI---------------'
#[Q,pol] = m.VI()

#print '-------------PI---------------'
[Q,pol] = m.PI()

#[V,pol] = m.QLearning(0.5)
#[V,pol] = m.RTDP()
#[V,pol] = m.RTDP2()

V = m.TD(pol)

#print Q

#print m.compare(V,Q,pol)

print(pol)

plotPolicy(m,np.transpose(np.array([pol])))
