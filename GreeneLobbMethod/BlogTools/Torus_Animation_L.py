import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
import copy
from matplotlib import gridspec


filename = "torus_L_animated.gif"
overlay = False

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
#moving the first point in a circle
while(cur < modulo/2):
        sample_pts[cur] = np.array([copy.deepcopy(data[0]),copy.deepcopy(data[(cur*2)%modulo])])
        cur = (cur + 1)
#moving the second point in a circle
while(cur < modulo):
        sample_pts[cur] = np.array([copy.deepcopy(data[(cur*2)%modulo]),copy.deepcopy(data[0])])
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
line1_data = np.array([[sample_L[0][0], sample_L[0][1]]])
line2_data = np.array([[sample_L[0][2], sample_L[0][3]]])
def update(num, pt1, pt2, L1, L2, ref1, ref2, line1, line2):
    global line1_data
    global line2_data

    line1_data = np.append(line1_data, np.array([[sample_L[num][0],sample_L[num][1]]]), axis=0)
    line2_data = np.append(line2_data, np.array([[sample_L[num][2],sample_L[num][3]]]), axis=0)
    line1.set_data(line1_data[:,0], line1_data[:,1])
    line2.set_data(line2_data[:,0], line2_data[:,1])
    
    pt1.set_data(sample_pts[num][0],sample_pts[num][1])
    pt2.set_data(sample_pts[num][2],sample_pts[num][3])
    L1.set_data(sample_L[num][0], sample_L[num][1])
    L2.set_data(sample_L[num][2], sample_L[num][3])
    ref1.set_data(sample_L[0][0], sample_L[0][1])
    ref2.set_data(sample_L[0][2], sample_L[0][3])

    

#################
#Graph of result#
#################

print("Rendering animation...")

gs = gridspec.GridSpec(2, 4)
plt.figure()
ax1 = plt.subplot(gs[0, 1:3])
ax2 = plt.subplot(gs[1, :2], )
ax3 = plt.subplot(gs[1, 2:])

cmap = 'winter'
cm = plt.get_cmap(cmap)

#lambdaXlambda plot
ax1.set_title("Pairs on Loop")
ax1.plot(data[:,0], data[:,1])

#L term 1 plot
print(L)
ax2.set_title("1st Term of L")
ax2.set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
ax2.plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    ax2.plot(L[i:i+2:,0],L[i:i+2:,1])

#L term 2 plot
ax3.set_title("2nd Term of L")
ax3.set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
if overlay:
    ax3.plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    ax3.plot(L[i:i+2:,2],L[i:i+2:,3])


#reference points

ref1, = ax2.plot(sample_L[0][0], sample_L[0][1], 'wo')
ref2, = ax3.plot(sample_L[0][2], sample_L[0][3], 'wo')

#the two points on lambdaXlambda
pt1, = ax1.plot(sample_pts[0][0],sample_pts[0][1], 'bo')
pt2, = ax1.plot(sample_pts[0][2],sample_pts[0][3], 'go')
#L term 1
L1, = ax2.plot(sample_L[0][0], sample_L[0][1], 'ko')
#L term 2
L2, = ax3.plot(sample_L[0][2], sample_L[0][3], 'ko')
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')

line1, = ax2.plot(line1_data[:,0], line1_data[:,1], 'r')
line2, = ax3.plot(line2_data[:,0], line2_data[:,1], 'r')

plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(6,8, forward=True)

print("   Animation initialization complete...")

ani = animation.FuncAnimation(fig, update, sample_pts.shape[0], fargs=(pt1, pt2, L1, L2, ref1, ref2, line1, line2,), blit=False)
ani.save(filename, writer='imagemagick')
plt.show()
