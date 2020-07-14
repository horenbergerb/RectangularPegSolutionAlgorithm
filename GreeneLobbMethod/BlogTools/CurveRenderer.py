import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
import copy

###################################
#Utilities and data initialization#
###################################


#Here we're actually trying to load up some data
print("Preparing Jordan curve data...")
data = None
try:
    data = np.loadtxt("customshape.txt")
    print("   Loaded curve from customshape.txt")
except:
    print("   Failed to load custom curve; go draw one!")
    quit()

#################
#Graph of result#
#################

fig, axs = plt.subplots(1)

#curve plot
axs.plot(data[:,0], data[:,1])
axs.set_aspect('equal')

plt.tight_layout()

plt.savefig('rawcurve.png')
