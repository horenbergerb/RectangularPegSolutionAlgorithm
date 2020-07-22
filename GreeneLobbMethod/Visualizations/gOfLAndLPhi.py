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

#this decides whether to impose the original curve on the second L and L_phi terms
#it's good to have this on during your first run, but turning it off shows the surface itself in more detail
overlay = True
#you can change the extension to gif if you want
filename = 'glandglphisurfaces.mp4'
#angle of the rectangle we're searching for in radians
phi = 3.14159/2.0

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
def get_phi(w):
    return np.arctan2(w[1], w[0])
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
def g(z,w):
    w_angle = get_phi(w)
    return np.concatenate((z.copy()/np.sqrt(2.0), add_phi(w.copy(), w_angle)))


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
print("   Preparing sample values for animation...")

#sample L data for our sample points
sample_L = copy.deepcopy(sample_pts)
for cur1 in range(0, sample_L.shape[0]):
    sample_L[cur1] = l(sample_L[cur1][0:2], sample_L[cur1][2:])

#sample g(L) data
sample_g_L = copy.deepcopy(sample_L)
for cur1 in range(0, sample_g_L.shape[0]):
    sample_g_L[cur1] = g(sample_g_L[cur1][0:2], sample_g_L[cur1][2:])

                          
#sample L_phi data for our sample points
sample_L_phi = copy.deepcopy(sample_L)
for cur1 in range(0, sample_L_phi.shape[0]):
    sample_L_phi[cur1] = R_phi(phi, sample_L_phi[cur1][0:2], sample_L_phi[cur1][2:])

#sample g(L_phi) data
sample_g_L_phi = copy.deepcopy(sample_L_phi)
for cur1 in range(0, sample_g_L.shape[0]):
    sample_g_L_phi[cur1] = g(sample_g_L_phi[cur1][0:2], sample_g_L_phi[cur1][2:])
    
                          
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
#L_phi as defined in the paper. we plug lambdaXlambda into l
L_phi = copy.deepcopy(L)
for cur1 in range(0, L.shape[0]):
    L_phi[cur1] = R_phi(phi, L_phi[cur1][0:2], L_phi[cur1][2:])

print("   Preparing total g(L)...")
#L as defined in the paper. we plug lambdaXlambda into l
g_L = copy.deepcopy(L)
for cur1 in range(0, g_L.shape[0]):
    g_L[cur1] = g(g_L[cur1][0:2], g_L[cur1][2:])

print("   Preparing total g(L_phi)...")
#L_phi as defined in the paper. we plug lambdaXlambda into l
g_L_phi = copy.deepcopy(L_phi)
for cur1 in range(0, L.shape[0]):
    g_L_phi[cur1] = g(g_L_phi[cur1][0:2], g_L_phi[cur1][2:])
    

#update function for the animator
def update(num, pt1, pt2, L1, L2, L_phi1, L_phi2, g_L1, g_L2, g_L_phi1, g_L_phi2):
    pt1.set_data(sample_pts[num][0],sample_pts[num][1])
    pt2.set_data(sample_pts[num][2],sample_pts[num][3])
    L1.set_data(sample_L[num][0], sample_L[num][1])
    L2.set_data(sample_L[num][2], sample_L[num][3])
    L_phi1.set_data(sample_L_phi[num][0], sample_L_phi[num][1])
    L_phi2.set_data(sample_L_phi[num][2], sample_L_phi[num][3])
    g_L1.set_data(sample_g_L[num][0], sample_g_L[num][1])
    g_L2.set_data(sample_g_L[num][2], sample_g_L[num][3])
    g_L_phi1.set_data(sample_g_L_phi[num][0], sample_g_L_phi[num][1])
    g_L_phi2.set_data(sample_g_L_phi[num][2], sample_g_L_phi[num][3])

#################
#Graph of result#
#################

print("Rendering animation...")

#this is a grid to make the first row have only one plot while the others have two
gs = gridspec.GridSpec(5, 4)
plt.figure()
#excuse the disgusting numbering; i'm lazy
ax1 = plt.subplot(gs[0, 1:3])

ax2 = plt.subplot(gs[1, :2], )
ax3 = plt.subplot(gs[1, 2:])

ax6 = plt.subplot(gs[2, :2], )
ax7 = plt.subplot(gs[2, 2:])

ax4 = plt.subplot(gs[3, :2], )
ax5 = plt.subplot(gs[3, 2:])

ax8 = plt.subplot(gs[4, :2], )
ax9 = plt.subplot(gs[4, 2:])

cmap = 'winter'
cm = plt.get_cmap(cmap)

print("   Initializing curve plot...")

#lambdaXlambda plot
ax1.set_title("Lambda Pairs")
ax1.set_prop_cycle('color',[cm(1.*i/(data.shape[0]-1)) for i in range(data.shape[0]-1)])
for i in range(data.shape[0]-1):
    ax1.plot(data[i:i+2:,0],data[i:i+2:,1])

print("   Initializing L plots...")
    
#L term 1 plot
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

print("   Initializing g(L) plots...")
    
#g_L term 1 plot
ax6.set_title("1st Term of g(L)")
ax6.set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
ax6.plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    ax6.plot(g_L[i:i+2:,0],g_L[i:i+2:,1])

#g_L term 2 plot
ax7.set_title("2nd Term of g(L)")
ax7.set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
if overlay:
    ax7.plot(data[:,0], data[:,1])
for i in range(L.shape[0]-1):
    ax7.plot(g_L[i:i+2:,2],g_L[i:i+2:,3])

    
print("   Initializing L_phi plots...")
    
#L_phi term 1 plot
ax4.set_title("1st Term of L_phi")
ax4.set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
ax4.plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    ax4.plot(L_phi[i:i+2:,0],L_phi[i:i+2:,1])

#L_phi term 2 plot
ax5.set_title("2nd Term of L_phi")
ax5.set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
if overlay:
    ax5.plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    ax5.plot(L_phi[i:i+2:,2],L_phi[i:i+2:,3])

print("   Initializing g(L_phi) plots...")
    
#L_phi term 1 plot
ax8.set_title("1st Term of g(L_phi)")
ax8.set_prop_cycle('color', [cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
ax8.plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    ax8.plot(g_L_phi[i:i+2:,0],g_L_phi[i:i+2:,1])

#L_phi term 2 plot
ax9.set_title("2nd Term of g(L_phi)")
ax9.set_prop_cycle('color',[cm(1.*i/(L.shape[0]-1)) for i in range(L.shape[0]-1)])
#an overlay of the original just to intuitively see what's goin on
if overlay:
    ax9.plot(data[:,0], data[:,1])
for i in range(L_phi.shape[0]-1):
    ax9.plot(g_L_phi[i:i+2:,2],g_L_phi[i:i+2:,3])


print("   Initializing animated points...")
    
#the two points on lambdaXlambda
pt1, = ax1.plot(sample_pts[0][0],sample_pts[0][1], 'go')
pt2, = ax1.plot(sample_pts[0][2],sample_pts[0][3], 'go')
#L term 1
L1, = ax2.plot(sample_L[0][0], sample_L[0][1], 'go')
#L term 2
L2, = ax3.plot(sample_L[0][2], sample_L[0][3], 'go')
#L_phi term 1
L_phi1, = ax4.plot(sample_L_phi[0][0], sample_L_phi[0][1], 'go')
#L_phi term 2
L_phi2, = ax5.plot(sample_L_phi[0][2], sample_L_phi[0][3], 'go')
#g_L term 1
g_L1, = ax6.plot(sample_g_L[0][0], sample_g_L[0][1], 'go')
#g_L term 2
g_L2, = ax7.plot(sample_g_L[0][2], sample_g_L[0][3], 'go')
#g_L_phi term 1
g_L_phi1, = ax8.plot(sample_g_L_phi[0][0], sample_g_L_phi[0][1], 'go')
#g_L_phi term 2
g_L_phi2, = ax9.plot(sample_g_L_phi[0][2], sample_g_L_phi[0][3], 'go')
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')
ax5.set_aspect('equal')
ax5.set_aspect('equal')
ax6.set_aspect('equal')
ax7.set_aspect('equal')
ax8.set_aspect('equal')
ax9.set_aspect('equal')

print("   Animation initialization complete...")
print("   Beginning actual animation process...")

fig = plt.gcf()
#use either this or tight_layout
#testing to see how they affect whitespace
plt.subplots_adjust(wspace=None, hspace=None)
#plt.tight_layout()
fig.set_size_inches(10,12, forward=True)

ani = animation.FuncAnimation(fig, update, sample_pts.shape[0], fargs=(pt1, pt2, L1, L2, L_phi1, L_phi2, g_L1, g_L2, g_L_phi1, g_L_phi2,), blit=False)
ani.save(filename, writer='ffmpeg')

print("Animation complete! Saved to {}".format(filename))

plt.show()
