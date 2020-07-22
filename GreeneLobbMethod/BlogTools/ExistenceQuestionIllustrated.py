import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
import copy
import matplotlib.gridspec as gridspec

############
#Parameters#
############

#desired rectangle angle
phi = 3.14159/4.0
#file output. change extension to gif or mp4 at your leisure
filename = 'rectangleexistence.png'

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

#does the rectangle construction from two complex numbers
#note: will only make an actual rectangle if the two points meet special conditions
def get_rectangle_pts(phi, pts):
    z = pts[0:2]
    w = pts[2:]
    pts = []
    pts.append(z+w)
    pts.append(z-w)
    pts.append(z+(add_phi(w,-phi)))
    pts.append(z-(add_phi(w,-phi)))
    return np.array(pts)
    
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
#samples for the first pair of animated points
#speeds of the two points
speed1 = 1
speed2 = 2
sample_pts1 = np.zeros((data.shape[0], 2,2))
modulo = data.shape[0]
cur = 0
while(cur < modulo):
        sample_pts1[cur] = np.array([copy.deepcopy(data[(cur*speed1)%modulo]),copy.deepcopy(data[(cur*speed2)%modulo])])
        cur = (cur + 1)
#we flatten into 4 vectors rather than 2 complex numbers
#it's easier to work with computationally
sample_pts1 = np.reshape(sample_pts1, (sample_pts1.shape[0],4))
print("   Preparing sample L values...")

#samples for the second pair of animated points
#offset of these points from first sample points
offset = 30
speed3 = 1
speed4 = 2
sample_pts2 = np.zeros((data.shape[0], 2,2))
modulo = data.shape[0]
cur = 0
while(cur < modulo):
        sample_pts2[cur] = np.array([copy.deepcopy(data[((cur+offset)*speed3)%modulo]),copy.deepcopy(data[((cur+offset)*speed4)%modulo])])
        cur = (cur + 1)
#we flatten into 4 vectors rather than 2 complex numbers
#it's easier to work with computationally
sample_pts2 = np.reshape(sample_pts2, (sample_pts2.shape[0],4))
print("   Preparing sample L values...")


#sample L data for our sample points
sample_L = copy.deepcopy(sample_pts1)
for cur1 in range(0, sample_L.shape[0]):
    sample_L[cur1] = l(sample_L[cur1][0:2], sample_L[cur1][2:])


#sample L_phi data for our sample points
sample_L_phi = copy.deepcopy(sample_pts1)
for cur1 in range(0, sample_L_phi.shape[0]):
    #run through l then R_phi
    sample_L_phi[cur1] = l(sample_L_phi[cur1][0:2], sample_L_phi[cur1][2:])
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
def update(num,sample1_pt1, sample1_pt2, L1, L2, L_phi1, L_phi2, L_rectangle, L_phi_rectangle):
    sample1_pt1.set_data(sample_pts1[num][0],sample_pts1[num][1])
    sample1_pt2.set_data(sample_pts1[num][2],sample_pts1[num][3])
    L1.set_data(sample_L[num][0], sample_L[num][1])
    L2.set_data(sample_L[num][2], sample_L[num][3])
    L_phi1.set_data(sample_L_phi[num][0], sample_L_phi[num][1])
    L_phi2.set_data(sample_L_phi[num][2], sample_L_phi[num][3])
    cur_rec = get_rectangle_pts(phi, copy.deepcopy(sample_L[num]))
    L_rectangle.set_data(cur_rec[:,0], cur_rec[:,1])
    cur_rec = get_rectangle_pts(phi, copy.deepcopy(sample_L_phi[num]))
    L_phi_rectangle.set_data(cur_rec[:,0], cur_rec[:,1])
    
#################
#Graph of result#
#################

print("Rendering animation...")

gs = gridspec.GridSpec(3, 4)
plt.figure()
ax1 = plt.subplot(gs[0, :2], )
#ignore this painful numbering
ax6 = plt.subplot(gs[0, 2:])
ax2 = plt.subplot(gs[1, :2], )
ax3 = plt.subplot(gs[1, 2:])
ax4 = plt.subplot(gs[2, :2], )
ax5 = plt.subplot(gs[2, 2:])

#second lambdaXlambda plot
ax1.set_title("Pair One")
ax1.plot(data[:,0], data[:,1])

#second lambdaXlambda plot
ax6.set_title("Pair Two")
ax6.plot(data[:,0], data[:,1])
    
#L plot
ax2.set_title("Pair One 1st Rectangle Intermediate")
#an overlay of the original just to intuitively see what's goin on
ax2.plot(data[:,0], data[:,1])

#L_phi plot
ax3.set_title("Pair Two 2nd Rectangle Intermediate")
#an overlay of the original just to intuitively see what's goin on
ax3.plot(data[:,0], data[:,1])

#rectangle from L points
ax4.set_title("Pair One 1st Rectangle")
ax4.plot(data[:,0], data[:,1])

ax5.set_title("Pair Two 2nd Rectangle")
ax5.plot(data[:,0], data[:,1])

#the two points on lambdaXlambda
pair1 = np.array([356., 294., 400., 346.])
pair2 = np.array([344.05, 317.17, 411.94, 322.828])
sample1_pt1, = ax1.plot([356., 400.], [294., 346.], 'go')
sample1_pt2, = ax6.plot([344.05, 411.94],[317.17, 322.828], 'ko')
#L term 1
L = l(pair1[:2], pair1[2:])
L1, = ax2.plot(L[0], L[1], 'go')
#L term 2
L2, = ax2.plot(L[2], L[3], 'ko')
#L_phi term 1
L_phi = l(pair2[:2], pair2[2:])
L_phi = R_phi(phi, L_phi[:2], L_phi[2:])
L_phi1, = ax3.plot(L_phi[0], L_phi[1], 'go')
#L_phi term 2
L_phi2, = ax3.plot(L_phi[2], L_phi[3], 'ko')

#rectangles for both L and L_phi

cur_rec = get_rectangle_pts(phi, copy.deepcopy(L))
print(cur_rec)
L_rectangle, = ax4.plot(cur_rec[:,0], cur_rec[:,1], 'go')

cur_rec = get_rectangle_pts(phi, copy.deepcopy(L_phi))
print(cur_rec)
L_phi_rectangle, = ax5.plot(cur_rec[:,0], cur_rec[:,1], 'go')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')
ax5.set_aspect('equal')
ax6.set_aspect('equal')

plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(10,8, forward=True)

print("   Animation initialization complete...")

plt.savefig("existenceclaimexample.png")
plt.show()
