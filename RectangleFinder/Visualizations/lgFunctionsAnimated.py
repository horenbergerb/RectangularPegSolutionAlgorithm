import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
import copy

#######
#TO DO#
#######
#add a second pair of lambda points in the [0][1] plot
  #wire these to the L_phi plots

#fix plots on row 4 which are supposed to show the animated L and L_phi dots with a line connecting the pairs 
  
#add 2 plots showing the "rectangles" that would be constructed with the pairs from L and L_phi
  #this is in progress


phi = 3.14159/4.0

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

#utilities
def to_polar(data):
    mag = np.sqrt((data[0]**2)+(data[1]**2))
    angle = np.arctan2(data[1],data[0])
    return np.array([mag, angle])
def to_cartesian(data):
    x = data[0]*np.cos(data[1])
    y = data[0]*np.sin(data[1])
    return np.array([x,y])
def add_phi(data, phi):
    data = to_polar(data)
    data[1] += phi
    data = to_cartesian(data)
    return data

    
#So the first thing is the definition of these l and g functions
#warning: must input in cartesian coords
def l(z,w):
    return np.concatenate(((z.copy()+w.copy())/2.0, (z.copy()-w.copy())/2.0))
#defining R_phi as in the paper
def R_phi(phi, z, w):
    w = to_polar(w)
    w[1] = w[1]+phi
    w = to_cartesian(w)
    return np.concatenate((z, w))


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

#sample L data for our sample points
sample_L_phi = copy.deepcopy(sample_L)
for cur1 in range(0, sample_L_phi.shape[0]):
    sample_L_phi[cur1] = R_phi(phi,sample_L_phi[cur1][0:2], sample_L_phi[cur1][2:])
    
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

print("   Preparing total L_phi...")
#L as defined in the paper. we plug lambdaXlambda into l
L_phi = copy.deepcopy(L)
for cur1 in range(0, L_phi.shape[0]):
    L_phi[cur1] = R_phi(phi, L_phi[cur1][0:2], L_phi[cur1][2:])


#update function for the animator
def update(num, pt1, pt2, L1, L2, L_phi1, L_phi2, term1_line, term2_line):
    pt1.set_data(sample_pts[num][0],sample_pts[num][1])
    pt2.set_data(sample_pts[num][2],sample_pts[num][3])
    L1.set_data(sample_L[num][0], sample_L[num][1])
    L2.set_data(sample_L[num][2], sample_L[num][3])
    L_phi1.set_data(sample_L_phi[num][0], sample_L_phi[num][1])
    L_phi2.set_data(sample_L_phi[num][2], sample_L_phi[num][3])
    term1_line.set_data([sample_L[num][0], sample_L_phi[num][0]], [sample_L[num][1], sample_L_phi[num][0]])
    term2_line.set_data([sample_L[num][2], sample_L_phi[num][2]], [sample_L[num][3], sample_L_phi[num][3]])
    
#################
#Graph of result#
#################

print("Rendering animation...")

fig, axs = plt.subplots(5, 2, figsize=(14,12))
fig.suptitle("Lambda Pairs vs. Terms of L vs. Terms of L_phi")

cmap = 'winter'
cm = plt.get_cmap(cmap)

#lambdaXlambda plot
axs[0][0].set_title("Lambda Pairs")
axs[0][0].set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[0][0].plot(data[i:i+2:,0],data[i:i+2:,1])

#L term 1 plot
axs[1][0].set_title("Corresponding 1st Term of L")
axs[1][0].set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[1][0].plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    axs[1][0].plot(L[i:i+2:,0],L[i:i+2:,1])

#L term 2 plot
axs[1][1].set_title("Corresponding 2nd Term of L")
axs[1][1].set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[1][1].plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    axs[1][1].plot(L[i:i+2:,2],L[i:i+2:,3])

#L_phi term 1 plot
axs[2][0].set_title("Corresponding 1st Term of L_phi")
axs[2][0].set_prop_cycle('color', [cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[2][0].plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    axs[2][0].plot(L_phi[i:i+2:,0],L_phi[i:i+2:,1])

#L_phi term 2 plot
axs[2][1].set_title("Corresponding 2nd Term of L_phi")
axs[2][1].set_prop_cycle('color',[cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[2][1].plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    axs[2][1].plot(L_phi[i:i+2:,2],L_phi[i:i+2:,3])

#term_1 overlap plot
axs[3][0].set_title("Overlap of L and L_phi first terms")
axs[3][0].set_prop_cycle('color', [cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])

#term_2 overlap plot
axs[3][1].set_title("Overlap of L and L_phi 2nd terms")
axs[3][1].set_prop_cycle('color',[cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])

#rectangle from L points
axs[4][0].set_title("Rectangle from L terms")
axs[4][1].set_title("Rectangle from L_phi terms")
    
#the two points on lambdaXlambda
pt1, = axs[0][0].plot(sample_pts[0][0],sample_pts[0][1], 'go')
pt2, = axs[0][0].plot(sample_pts[0][2],sample_pts[0][3], 'go')
#L term 1
L1, = axs[1][0].plot(sample_L[0][0], sample_L[0][1], 'go')
#L term 2
L2, = axs[1][1].plot(sample_L[0][2], sample_L[0][3], 'go')
#L_phi term 1
L_phi1, = axs[2][0].plot(sample_L_phi[0][0], sample_L_phi[0][1], 'go')
#L_phi term 2
L_phi2, = axs[2][1].plot(sample_L_phi[0][2], sample_L_phi[0][3], 'go')

#For the dots alone
#term 1
term1_line, = axs[3][0].plot([sample_L[0][0], sample_L_phi[0][0]], [sample_L[0][1], sample_L_phi[0][0]], 'g')
#term2
term2_line, = axs[3][1].plot([sample_L[0][2], sample_L_phi[0][2]], [sample_L[0][3], sample_L_phi[0][3]], 'g')

#rectangles for both L and L_phi
L_rectangle, = axs[4][0].plot([sample_L[0][0] + sample_L[0][2], sample_L[0][0] - sample_L[0][2], sample_L[0][0] + R_phi(-phi, sample_L[0][0:2], sample_L[0][2:])[2]), sample_L[0][0] - R_phi(-phi, sample_L_phi[0][0:2], sample_L_phi[0][2:])[2])], [sample_L[0][1] + sample_L[0][4], sample_L[0][1] - sample_L[0][4], sample_L[0][1] + R_phi(-phi, sample_L[0][0:2], sample_L[0][2:])[4]), sample_L[0][1] - R_phi(-phi, sample_L_phi[0][0:2], sample_L_phi[0][2:])[4])], 'go'))

L_phi_rectangle, = axs[4][0].plot([sample_L[0][0] + sample_L[0][2], sample_L[0][0] - sample_L[0][2], sample_L[0][0] + R_phi(-phi, sample_L[0][0:2], sample_L[0][2:])[2]), sample_L[0][0] - R_phi(-phi, sample_L_phi[0][0:2], sample_L_phi[0][2:])[2])], [sample_L[0][1] + sample_L[0][4], sample_L[0][1] - sample_L[0][4], sample_L[0][1] + R_phi(-phi, sample_L[0][0:2], sample_L[0][2:])[4]), sample_L[0][1] - R_phi(-phi, sample_L_phi[0][0:2], sample_L_phi[0][2:])[4])], 'go'))





for x in range(0,3):
    for y in range(0,2):
        axs[x][y].set_aspect('equal')

plt.tight_layout()

print("   Animation initialization complete...")

ani = animation.FuncAnimation(fig, update, sample_pts.shape[0], fargs=(pt1, pt2, L1, L2, L_phi1, L_phi2, term1_line, term2_line,), blit=False)
ani.save('pairsandfirstlterm.gif', writer='imagemagick')
plt.show()
