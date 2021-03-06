import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import animation
import copy

############
#Parameters#
############

#desired rectangle angle
phi = 3.14159/4.0
#file output. change extension to gif or mp4 at your leisure
filename = 'Lswappedpoints.png'
#overlays the original curve on the second terms' plots. good for intuition but zooms it out too far
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
    sample_L_phi[cur1] = l(sample_L_phi[cur1][2:], sample_L_phi[cur1][0:2])
    
print("   Preparing lambdaXlambda...")
#defining lambdaXlambda
lambda_cross_lambda = np.zeros((data.shape[0], data.shape[0], 2,2))
for cur1 in range(0, data.shape[0]):
    for cur2 in range(0, data.shape[0]):
        lambda_cross_lambda[cur1][cur2] = np.array([copy.deepcopy(data[cur1]),copy.deepcopy(data[cur2])])
#we flatten into 4 vectors rather than 2 complex numbers
#it's easier to work with computationally
lambda_cross_lambda = np.reshape(lambda_cross_lambda, (lambda_cross_lambda.shape[0]*lambda_cross_lambda.shape[1], lambda_cross_lambda.shape[2]*lambda_cross_lambda.shape[3]))

print("   Preparing pair one L...")
#L as defined in the paper. we plug lambdaXlambda into l
L = copy.deepcopy(lambda_cross_lambda)
for cur1 in range(0, L.shape[0]):
    L[cur1] = l(L[cur1][0:2], L[cur1][2:])

print("   Preparing pair 2 L...")
#L as defined in the paper. we plug lambdaXlambda into l
L_phi = copy.deepcopy(lambda_cross_lambda)
for cur1 in range(0, L_phi.shape[0]):
    L_phi[cur1] = l(L_phi[cur1][2:], L_phi[cur1][0:2])

    
#################
#Graph of result#
#################

fig, axs = plt.subplots(1, 3)


axs[0].plot(data[:,0],data[:,1])
#axs[1].plot(data[:,0],data[:,1])
#axs[2].plot(data[:,0],data[:,1])

axs[0].set_title("Pair of points")

axs[1].set_title("Corresponding L")
axs[2].set_title("Corresponding L for flipped points")

#rectangle from L points
#the two points on lambdaXlambda
sample1_pt1, = axs[0].plot(sample_pts1[30][0],sample_pts1[30][1], 'go')
sample1_pt2, = axs[0].plot(sample_pts1[30][2],sample_pts1[30][3], 'go')



#L_plot1, = axs[1].plot(sample_L[30][0], sample_L[30][1], 'go')
L_plot2, = axs[1].plot(sample_L[30][2], sample_L[30][3], 'ko')

#L_phi_plot1, = axs[2].plot(sample_L_phi[30][0], sample_L_phi[30][1], 'go')
L_phi_plot2, = axs[2].plot(sample_L_phi[30][2], sample_L_phi[30][3], 'ko')

for x in range(0,3):
    axs[x].set_aspect('equal')

plt.tight_layout()

plt.show()
