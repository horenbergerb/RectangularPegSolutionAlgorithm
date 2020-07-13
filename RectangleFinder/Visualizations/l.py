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

#So the first thing is the definition of these l and g functions
#warning: must input in cartesian coords
def l(z,w):
    return np.concatenate(((z.copy()+w.copy())/2.0, (z.copy()-w.copy())/2.0))

#defining lambdaXlambda
lambda_cross_lambda = np.zeros((data.shape[0], data.shape[0], 2,2))
for cur1 in range(0, data.shape[0]):
    for cur2 in range(0, data.shape[0]):
        lambda_cross_lambda[cur1][cur2] = np.array([copy.deepcopy(data[cur1]),copy.deepcopy(data[cur2])])
#we flatten into 4 vectors rather than 2 complex numbers
#it's easier to work with computationally
lambda_cross_lambda = np.reshape(lambda_cross_lambda, (lambda_cross_lambda.shape[0]*lambda_cross_lambda.shape[1], lambda_cross_lambda.shape[2]*lambda_cross_lambda.shape[3]))

print("   Preparing L...")
#L as defined in the paper. we plug lambdaXlambda into l
L = copy.deepcopy(lambda_cross_lambda)
for cur1 in range(0, L.shape[0]):
    L[cur1] = l(L[cur1][0:2], L[cur1][2:])


#update function for the animator
def update(num, pt1, pt2, out):
    pt1.set_data(lambda_cross_lambda[num][0],lambda_cross_lambda[num][1])
    pt2.set_data(lambda_cross_lambda[num][2],lambda_cross_lambda[num][3])
    out.set_data(L[num][0], L[num][1])

#################
#Graph of result#
#################

print("Rendering animation...")

fig, axs = plt.subplots(2)
fig.suptitle("Lambda Pairs vs. First Term of L, i.e. Midpoints")

cmap = 'winter'
cm = plt.get_cmap(cmap)
axs[0].set_title("Lambda Pairs")
axs[0].set_color_cycle([cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[0].plot(data[i:i+2:,0],data[i:i+2:,1])
#axs[0].plot(data[:,0], data[:,1])
axs[1].set_title("Corresponding First Term of L, i.e. Midpoint")
axs[1].set_color_cycle([cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
for i in range(L.shape[0]-1):
    axs[1].plot(L[i:i+2:,0],L[i:i+2:,1])

#axs[1].plot(L[:,0], L[:,1])

pt1, = axs[0].plot(lambda_cross_lambda[0][0],lambda_cross_lambda[0][1], 'go')
pt2, = axs[0].plot(lambda_cross_lambda[0][2],lambda_cross_lambda[0][3], 'go')
out, = axs[1].plot(L[0][0], L[0][1], 'go')
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')

ani = animation.FuncAnimation(fig, update, 100, fargs=(pt1, pt2, out,), blit=False)
ani.save('pairsandfirstlterm.gif', writer='imagemagick')
plt.show()
