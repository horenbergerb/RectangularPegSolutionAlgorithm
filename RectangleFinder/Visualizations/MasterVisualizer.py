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
phi = 3.14159/2.0
#file output. change extension to gif or mp4 at your leisure
filename = 'mastervisualizer.mp4'
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
sample_L_phi = copy.deepcopy(sample_pts2)
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
def update(num,sample1_pt1, sample1_pt2, sample2_pt1, sample2_pt2, L1, L2, L_phi1, L_phi2, term1_line, term2_line, L_rectangle, L_phi_rectangle):
    sample1_pt1.set_data(sample_pts1[num][0],sample_pts1[num][1])
    sample1_pt2.set_data(sample_pts1[num][2],sample_pts1[num][3])
    sample2_pt1.set_data(sample_pts2[num][0],sample_pts2[num][1])
    sample2_pt2.set_data(sample_pts2[num][2],sample_pts2[num][3])
    L1.set_data(sample_L[num][0], sample_L[num][1])
    L2.set_data(sample_L[num][2], sample_L[num][3])
    L_phi1.set_data(sample_L_phi[num][0], sample_L_phi[num][1])
    L_phi2.set_data(sample_L_phi[num][2], sample_L_phi[num][3])
    term1_line.set_data([sample_L[num][0], sample_L_phi[num][0]], [sample_L[num][1], sample_L_phi[num][1]])
    term2_line.set_data([sample_L[num][2], sample_L_phi[num][2]], [sample_L[num][3], sample_L_phi[num][3]])
    cur_rec = get_rectangle_pts(phi, copy.deepcopy(sample_L[num]))
    L_rectangle.set_data(cur_rec[:,0], cur_rec[:,1])
    cur_rec = get_rectangle_pts(phi, copy.deepcopy(sample_L_phi[num]))
    L_phi_rectangle.set_data(cur_rec[:,0], cur_rec[:,1])
    
#################
#Graph of result#
#################

print("Rendering animation...")

fig, axs = plt.subplots(5, 2, figsize=(14,12))

cmap = 'winter'
cm = plt.get_cmap(cmap)

#second lambdaXlambda plot
axs[0][0].set_title("Lambda Pair for L")
axs[0][0].set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[0][0].plot(data[i:i+2:,0],data[i:i+2:,1])

#second lambdaXlambda plot
axs[0][1].set_title("Lambda Pair for L_phi")
axs[0][1].set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[0][1].plot(data[i:i+2:,0],data[i:i+2:,1])
    
#L term 1 plot
axs[1][0].set_title("1st Term of L")
axs[1][0].set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[1][0].plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    axs[1][0].plot(L[i:i+2:,0],L[i:i+2:,1])

#L term 2 plot
axs[1][1].set_title("2nd Term of L")
axs[1][1].set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
if overlay:
    axs[1][1].plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    axs[1][1].plot(L[i:i+2:,2],L[i:i+2:,3])

#L_phi term 1 plot
axs[2][0].set_title("1st Term of L_phi")
axs[2][0].set_prop_cycle('color', [cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
axs[2][0].plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    axs[2][0].plot(L_phi[i:i+2:,0],L_phi[i:i+2:,1])

#L_phi term 2 plot
axs[2][1].set_title("2nd Term of L_phi")
axs[2][1].set_prop_cycle('color',[cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
if overlay:
    axs[2][1].plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    axs[2][1].plot(L_phi[i:i+2:,2],L_phi[i:i+2:,3])

#term_1 overlap plot
axs[3][0].set_title("Overlap of L, L_phi 1st terms")
#axs[3][0].set_prop_cycle('color', [cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])
#setting the scale properly
test_pts_x = np.concatenate((sample_L[:,0], sample_L_phi[:,0]))
test_pts_y = np.concatenate((sample_L[:,1], sample_L_phi[:,1]))
axs[3][0].set_xlim([np.amin(test_pts_x)-10, np.amax(test_pts_x)+10])
axs[3][0].set_ylim([np.amin(test_pts_y)-10, np.amax(test_pts_y)+10])

#term_2 overlap plot
axs[3][1].set_title("Overlap of L, L_phi 2nd terms")
#axs[3][1].set_prop_cycle('color',[cm(1.*i/(L_phi.shape[0]-1)) for i in range(L_phi.shape[0]-1)])
#setting the scale properly
test_pts_x = np.concatenate((sample_L[:,2], sample_L_phi[:,2]))
test_pts_y = np.concatenate((sample_L[:,3], sample_L_phi[:,3]))
axs[3][1].set_xlim([np.amin(test_pts_x)-10, np.amax(test_pts_x)+10])
axs[3][1].set_ylim([np.amin(test_pts_y)-10, np.amax(test_pts_y)+10])


#rectangle from L points
axs[4][0].set_title("Rectangle from L terms")
axs[4][0].set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[4][0].plot(data[i:i+2:,0],data[i:i+2:,1])
axs[4][1].set_title("Rectangle from L_phi terms")
axs[4][1].set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    axs[4][1].plot(data[i:i+2:,0],data[i:i+2:,1])
#the two points on lambdaXlambda
sample1_pt1, = axs[0][0].plot(sample_pts1[0][0],sample_pts1[0][1], 'go')
sample1_pt2, = axs[0][0].plot(sample_pts1[0][2],sample_pts1[0][3], 'go')
sample2_pt1, = axs[0][1].plot(sample_pts2[0][0],sample_pts2[0][1], 'go')
sample2_pt2, = axs[0][1].plot(sample_pts2[0][2],sample_pts2[0][3], 'go')
#L term 1
L1, = axs[1][0].plot(sample_L[0][0], sample_L[0][1], 'go')
#L term 2
L2, = axs[1][1].plot(sample_L[0][2], sample_L[0][3], 'go')
#L_phi term 1
L_phi1, = axs[2][0].plot(sample_L_phi[0][0], sample_L_phi[0][1], 'go')
#L_phi term 2
L_phi2, = axs[2][1].plot(sample_L_phi[0][2], sample_L_phi[0][3], 'go')

#For the overlap plots
#term 1
term1_line, = axs[3][0].plot([sample_L[0][0], sample_L_phi[0][0]], [sample_L[0][1], sample_L_phi[0][1]], 'g')
#term2
term2_line, = axs[3][1].plot([sample_L[0][2], sample_L_phi[0][2]], [sample_L[0][3], sample_L_phi[0][3]], 'g')

#rectangles for both L and L_phi

cur_rec = get_rectangle_pts(phi, copy.deepcopy(sample_L[0]))
print(cur_rec)
L_rectangle, = axs[4][0].plot(cur_rec[:,0], cur_rec[:,1], 'go')

cur_rec = get_rectangle_pts(phi, copy.deepcopy(sample_L_phi[0]))
print(cur_rec)
L_phi_rectangle, = axs[4][1].plot(cur_rec[:,0], cur_rec[:,1], 'go')

for x in range(0,5):
    for y in range(0,2):
        axs[x][y].set_aspect('equal')

plt.tight_layout()

print("   Animation initialization complete...")

ani = animation.FuncAnimation(fig, update, sample_pts1.shape[0], fargs=(sample1_pt1, sample1_pt2, sample2_pt1, sample2_pt2, L1, L2, L_phi1, L_phi2, term1_line, term2_line, L_rectangle, L_phi_rectangle,), blit=False)
ani.save(filename, writer='ffmpeg')
plt.show()
