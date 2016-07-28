import numpy as np
import lal
import math
import time

nsamps = 10240000

theta = np.linspace(0, 2*math.pi, nsamps)
phi = np.linspace(0, 2*math.pi, nsamps)

selected_modes = [(2,-2), (2,0), (2,2)]

def call_lalfunc(Lmax, theta, phi, selected_modes=None): 
	
	for l in range(2, Lmax+1):
		for m in range(-l, l+1):
			if selected_modes is not None and (l,m) not in selected_modes:
				continue
			for i in xrange(0, len(theta)):
				lal.SpinWeightedSphericalHarmonic(theta[i], phi[i], -2, l, m)
	return

start = time.clock()
call_lalfunc(2, theta, phi, selected_modes)
