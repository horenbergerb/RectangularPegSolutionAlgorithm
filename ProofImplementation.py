import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import simps
from scipy.integrate import quad
import copy
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import constants 

#By Beau Horenberger
#This is an implementation of some of the features which appear in the proof of the first theorem in the following paper:
#https://arxiv.org/pdf/2005.09193.pdf
#It's a pretty exciting proof, so I thought I might try to visualize what's going on

pi = constants.pi

#The interval between samples on our curve
curve_intervals = .07e-1

#margin for checking whether solutions are on lambdaX{0}
#can sometimes be tweaked to fix results instead of curve_interval
#better to have this small, but you'll also need small curve_interval
rtol = 1e-5
atol = 8e-2

#picking a phi
phi = pi/3.0

#Creating data for a simple circle (Jordan curve) to try and algorithmically solve for a rectangle inscription
def curve_data():
    #An assortment of sample curves
    #curve = np.array([np.array([np.sin(i),np.cos(i)]) for i in np.arange(0.0,2.0*pi, curve_intervals)])
    #curve = np.array([np.array([3.0*np.sin(i),np.cos(i)]) for i in np.arange(0.0,2.0*pi, curve_intervals)])
    curve = np.array([np.array([np.real((np.sin(2.0*i)+2.0)*np.exp(i*1j)),np.imag((np.sin(2.0*i)+2.0)*np.exp(i*1j))]) for i in np.arange(-0.0,2.0*pi, curve_intervals)])
    return curve

#Coordinate conversion functions which take [x,y] and [mag,angle]
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

print("Hello! The Jordan curve is chosen in the curve_data function, and phi is set in the code to {}".format(phi))
print("We're going to solve the points on the curve with the diagonal of angle phi")

#Jordan curve data
print("Preparing Jordan curve data...")
data = None
try:
    data = np.loadtxt("customshape.txt")
    print("Loaded curve from customshape.txt")
except:
    print("Failed to load custom curve; using mathematically generated curve")
    data = curve_data()

############################
#Begin Proof Implementation#
############################

#So the first thing is the definition of these l and g functions
#warning: must input in cartesian coords
def l(z,w):
    return np.array([(z.copy()+w.copy())/2.0, (z.copy()-w.copy())/2.0])
    
#warning: must input in cartesian coords
#we convert to polar inside
def g(z,w):
    w = to_polar(w)
    w[0] = w[0]/np.sqrt(2.0)
    w[1] = w[1]*2.0
    w = to_cartesian(w)
    return np.array([z, w])

print("Preparing lambdaXlambda...")
#Defining lambdaXlambda
#very janky tensor product
lambda_cross_lambda = np.zeros((data.shape[0], data.shape[0], 2,2))
for cur1 in range(0, data.shape[0]):
    for cur2 in range(0, data.shape[0]):
        lambda_cross_lambda[cur1][cur2] = np.array([copy.deepcopy(data[cur1]),copy.deepcopy(data[cur2])])

print("Preparing L...")
#L as defined in the paper. we plug lambdaXlambda into l
L = copy.deepcopy(lambda_cross_lambda)
for cur1 in range(0, data.shape[0]):
    for cur2 in range(0, data.shape[0]):
        #i use the splat operator, *, to unpack the list as arguments
        L[cur1][cur2] = l(*L[cur1][cur2])

#defining R_phi as in the paper
def R_phi(phi, z, w):
    w = to_polar(w)
    w[1] = w[1]+phi
    w = to_cartesian(w)
    return np.array([z, w])

print("Preparing L_phi...")
#creating L_phi as in the paper
L_phi = copy.deepcopy(L)
for cur1 in range(0, L_phi.shape[0]):
    for cur2 in range(0, L_phi.shape[1]):
        L_phi[cur1][cur2] = R_phi(phi, *L_phi[cur1][cur2])

print("Preparing M...")
#creating M as in the paper
M = copy.deepcopy(L)
for cur1 in range(0, L.shape[0]):
    for cur2 in range(0, L.shape[1]):
        M[cur1][cur2] = g(*M[cur1][cur2])

print("Preparing M_phi...")
#creating M_phi as in the paper
M_phi = copy.deepcopy(L_phi)
for cur1 in range(0, L_phi.shape[0]):
    for cur2 in range(0, L_phi.shape[1]):
        #observe we are using L_phi in this loop
        M_phi[cur1][cur2] = g(*M_phi[cur1][cur2])

#QUESTION: do we need to define the tangent spaces, T_p(L) and T_p(L_phi)

#According to the proof, we're done if we calculate
#the intersection points of M and M_phi or L and L_phi
#but this is an inefficient approximation can we use the intermediate proofs to speed it up?
def find_intersection_point(set1, set2):
    #tensor set1 and set2 but with scalar norms instead of 2 complex values as entries
    distances = np.zeros((set1.shape[0], set1.shape[1], set2.shape[0], set2.shape[1]))
    print("   Finding distances between all points")
    #preemptively finding points in set1 and set2 which are on lambda and zero:
    #finding all the distances between points on M and points on M_phi
    set1_on_lambda = np.array([[0]*data.shape[0]]*data.shape[0])
    set1_on_zero = np.array([[0]*data.shape[0]]*data.shape[0])
    set2_on_lambda = np.array([[0]*data.shape[0]]*data.shape[0])
    set2_on_zero = np.array([[0]*data.shape[0]]*data.shape[0])
    for val1 in range(0, data.shape[0]):
        for val2 in range(0, data.shape[0]):
            #checking if our terms are lambda (to avoid lambdaX{0})
            for x in data:
                if np.isclose(np.linalg.norm(x-set1[val1][val2][0]),0.0, rtol=rtol, atol=atol):
                    set1_on_lambda[val1][val2] = 1
                if np.isclose(np.linalg.norm(x-set2[val1][val2][0]),0.0, rtol=rtol, atol=atol):
                    set2_on_lambda[val1][val2] = 1
                if set1_on_lambda[val1][val2] and set2_on_lambda[val1][val2]:
                    break
            #checking if our terms are zero
            if np.isclose(np.linalg.norm(set1[val1][val2][1]),0.0, rtol=rtol, atol=atol):
                set1_on_zero[val1][val2]=1
            if np.isclose(np.linalg.norm(set2[val1][val2][1]),0.0, rtol=rtol, atol=atol):
                set2_on_zero[val1][val2]=1
                
    for M1 in range(0, data.shape[0]):
        for M2 in range(0, data.shape[0]):
            for M_phi1 in range(0, data.shape[0]):
                for M_phi2 in range(0, data.shape[0]):
                    #checking if second term is approximately zero (to avoid lambdaX{0}
                    if (set1_on_lambda[M1][M2] and set1_on_zero[M1][M2]) or (set2_on_lambda[M_phi1][M_phi2] and set2_on_zero[M_phi1][M_phi2]):
                        distances[M1][M2][M_phi1][M_phi2] = np.inf
                    #otherwise find the distance as usual
                    else:
                        distances[M1][M2][M_phi1][M_phi2] = np.linalg.norm(set1[M1][M2]-set2[M_phi1][M_phi2])
                            
    print("   Finding minimum distance...")
    #getting minimum distance
    dist_min = np.nanmin(distances)
    min_val_loc = np.where(distances == dist_min)
    print("   Minimum distance: {}".format(dist_min))
    return min_val_loc

#Solving the L and L_phi approach
print("Finding intersection point between L and L_phi points...")
min_val_loc_L = find_intersection_point(L, L_phi)
print("Minimum distance point in L:")
print(L[min_val_loc_L[0][0]][min_val_loc_L[1][0]])
print("Minimum distance point in L_phi:")
print(L_phi[min_val_loc_L[2][0]][min_val_loc_L[3][0]])
print("Minimum distance indices:")
print(min_val_loc_L)

print("")

L_intersection = copy.deepcopy(L[min_val_loc_L[0][0]][min_val_loc_L[1][0]])

L_solutions = np.array([L_intersection[0]+L_intersection[1], L_intersection[0]-L_intersection[1], L_intersection[0]+add_phi(L_intersection[1], -phi), L_intersection[0]-add_phi(L_intersection[1], -phi)])
print("Estimated intersection of L and L_phi: {}+{}i and {}+{}i".format(*np.concatenate((L_intersection[0], L_intersection[1]))))
print("Rectangle vertices from from L intersection")
print("   {}\n   {}\n   {}\n   {}".format(*L_solutions))


##########################
#End Proof Implementation#
##########################

#################
#Graph of result#
#################

fig = plt.figure()
ax = fig.add_subplot(111)
#Graphing solution from L and L_phi
#Lambda and solution points:
#plotting lambda
ax.set_title("L and L_phi Solution")
ax.plot(data[:,0], data[:,1])
#plotting the values for the intersection of L and L_phi
ax.scatter(L_intersection[:,0], L_intersection[:,1], color='red')
#plotting solution points
ax.scatter(L_solutions[:,0], L_solutions[:,1])
ax.set_aspect('equal')

plt.savefig("curvewithrectangle.png")
plt.show()


#######################
#THE REST OF THESE ARE#
#CURRENTLY BROKEN######
#######################
#I don't know how to handle plotting these high-dimensional functions
#All plotting code was before I had properly implemented the tensor product
#I'd like to plot more of the intermediate processes...
'''
#plotting (lambda,lambda) vs l(lambda, lambda)
fig,ax = plt.subplots(2,2)
for subplotouter in ax:
    for subplot in subplotouter:
        subplot.set_aspect('equal')

#setting up color scheme

        
fig.suptitle("Plugging LambdaXLambda into l")
ax[0][0].plot(data[:,0],data[:,01])
ax[0][0].set_title("Lambda")
ax[0][1].plot(L[:,0][:,0],L[:,0][:,1])
ax[0][1].set_title("First l(LambdaXLambda) Output")
ax[1][0].plot(data[:,0],data[:,01])
ax[1][0].set_title("Lambda")
ax[1][1].plot(L[:,1][:,0],L[:,1][:,1])
ax[1][1].set_title("Second l(LambdaXLambda) Output")
plt.show()
'''

#plotting L vs L_phi
'''
fig,ax = plt.subplots(2,2)
print(data)
for subplotouter in ax:
    for subplot in subplotouter:
        subplot.set_aspect('equal')

fig.suptitle("Plugging L into R_phi")
ax[0][0].plot(L[:,0][:,0],L[:,0][:,1])
ax[0][0].set_title("First L values")
ax[0][1].plot(L_phi[:,0][:,0],L_phi[:,0][:,1])
ax[0][1].set_title("First l(LambdaXLambda) Output")
ax[1][0].plot(L[:,0][:,0],L[:,0][:,1])
ax[1][0].set_title("Lambda")
ax[1][1].plot(L_phi[:,1][:,0],L_phi[:,1][:,1])
ax[1][1].set_title("Second l(LambdaXLambda) Output")
plt.show()
'''
