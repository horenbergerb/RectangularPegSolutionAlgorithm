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

print("Preparing variables...")
print("   Preparing sample points...")
#samples for the animated points
sample_pts = np.zeros((data.shape[0], 2,2))
modulo = data.shape[0]
cur = 0
while(cur < modulo):
        sample_pts[cur] = np.array([copy.deepcopy(data[cur]),copy.deepcopy(data[(cur*2)%modulo])])
        cur = (cur + 1)
#we flatten into 4 vectors rather than 2 complex numbers
#it's easier to work with computationally
sample_pts = np.reshape(sample_pts, (sample_pts.shape[0],4))
print("   Preparing sample L values...")

#sample L data for our sample points
sample_L = copy.deepcopy(sample_pts)
for cur1 in range(0, sample_L.shape[0]):
    sample_L[cur1] = l(sample_L[cur1][0:2], sample_L[cur1][2:])

print("   Preparing lambdaXlambda...")
#defining lambdaXlambda
lambda_cross_lambda = np.zeros((data.shape[0], data.shape[0], 2,2))
for cur1 in range(0, data.shape[0]):
    for cur2 in range(0, data.shape[0]):
        lambda_cross_lambda[cur1][cur2] = np.array([copy.deepcopy(data[cur1]),copy.deepcopy(data[cur2])])
#we flatten into 4 vectors rather than 2 complex numbers
#it's easier to work with computationally
lambda_cross_lambda = np.reshape(lambda_cross_lambda, (lambda_cross_lambda.shape[0]*lambda_cross_lambda.shape[1], lambda_cross_lambda.shape[2]*lambda_cross_lambda.shape[3]))

print("   Preparing total L...")
#L as defined in the paper. we plug lambdaXlambda into l
L = copy.deepcopy(lambda_cross_lambda)
for cur1 in range(0, L.shape[0]):
    L[cur1] = l(L[cur1][0:2], L[cur1][2:])


#update function for the animator
def update(num, pt1, pt2, L1, L2):
    pt1.set_data(sample_pts[num][0],sample_pts[num][1])
    pt2.set_data(sample_pts[num][2],sample_pts[num][3])
    L1.set_data(sample_L[num][0], sample_L[num][1])
    L2.set_data(sample_L[num][2], sample_L[num][3])

#################
#Graph of result#
#################

print("Rendering animation...")

fig, axs = plt.subplots(3)
fig.suptitle("Lambda Pairs vs. First Term of L, i.e. Midpoints")

cmap = 'winter'
cm = plt.get_cmap(cmap)

#lambdaXlambda plot
axs[0].set_title("Lambda Pairs")
axs[0].set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[0].plot(data[i:i+2:,0],data[i:i+2:,1])

#L term 1 plot
print(L)
axs[1].set_title("Corresponding 1st Term of L, i.e. Midpoint")
axs[1].set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[1].plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    axs[1].plot(L[i:i+2:,0],L[i:i+2:,1])

#L term 2 plot
axs[2].set_title("Corresponding 2nd Term of L, i.e. Midpoint")
axs[2].set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[2].plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    axs[2].plot(L[i:i+2:,2],L[i:i+2:,3])

#the two points on lambdaXlambda
pt1, = axs[0].plot(sample_pts[0][0],sample_pts[0][1], 'go')
pt2, = axs[0].plot(sample_pts[0][2],sample_pts[0][3], 'go')
#L term 1
L1, = axs[1].plot(sample_L[0][0], sample_L[0][1], 'go')
#L term 2
L2, = axs[2].plot(sample_L[0][2], sample_L[0][3], 'go')
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
axs[2].set_aspect('equal')

plt.tight_layout()

print("   Animation initialization complete...")

ani = animation.FuncAnimation(fig, update, sample_pts.shape[0], fargs=(pt1, pt2, L1, L2,), blit=False)
ani.save('landlphisurfaces.mp4', writer='imagemagick')
plt.show()
