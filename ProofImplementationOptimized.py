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
    return np.concatenate(((z.copy()+w.copy())/2.0, (z.copy()-w.copy())/2.0))
    
#warning: must input in cartesian coords
#we convert to polar inside
def g(z,w):
    w = to_polar(w)
    w[0] = w[0]/np.sqrt(2.0)
    w[1] = w[1]*2.0
    w = to_cartesian(w)
    return np.array(np.concatenate((z, w)))

print("Preparing lambdaXlambda...")
#Defining lambdaXlambda
#very janky tensor product
lambda_cross_lambda = np.zeros((data.shape[0], data.shape[0], 2,2))
for cur1 in range(0, data.shape[0]):
    for cur2 in range(0, data.shape[0]):
        lambda_cross_lambda[cur1][cur2] = np.array([copy.deepcopy(data[cur1]),copy.deepcopy(data[cur2])])
#flattening into 4 vectors rather than 2 complex numbers
lambda_cross_lambda = np.reshape(lambda_cross_lambda, (lambda_cross_lambda.shape[0]*lambda_cross_lambda.shape[1], lambda_cross_lambda.shape[2]*lambda_cross_lambda.shape[3]))

print("Preparing L...")
#L as defined in the paper. we plug lambdaXlambda into l
L = copy.deepcopy(lambda_cross_lambda)
for cur1 in range(0, L.shape[0]):
    L[cur1] = l(L[cur1][0:2], L[cur1][2:])

#defining R_phi as in the paper
def R_phi(phi, z, w):
    w = to_polar(w)
    w[1] = w[1]+phi
    w = to_cartesian(w)
    return np.concatenate((z, w))

print("Preparing L_phi...")
#creating L_phi as in the paper
L_phi = copy.deepcopy(L)
for cur1 in range(0, L_phi.shape[0]):
    L_phi[cur1] = R_phi(phi, L_phi[cur1][0:2], L_phi[cur1][2:])

print("Preparing M...")
#creating M as in the paper
M = copy.deepcopy(L)
for cur1 in range(0, L.shape[0]):
    M[cur1] = g(M[cur1][0:2], M[cur1][2:])

print("Preparing M_phi...")
#creating M_phi as in the paper
M_phi = copy.deepcopy(L_phi)
for cur1 in range(0, L_phi.shape[0]):
    #observe we are using L_phi in this loop
    M_phi[cur1] = g(M_phi[cur1][0:2], M_phi[cur1][2:])

#since we can't pick a solution whose first term is on lambda and
#whose second term is zero, we classify all of our terms
#WARNING: set1 must be sorted before this
def find_lambda_or_zero(set1):
    set1_on_lambda = np.array([False]*set1.shape[0])
    set1_on_zero = np.array([False]*set1.shape[0])
    for set1_cur in range(set1.shape[0]):
        #checking if points are on lambda
        for x in data:
            if np.isclose(np.linalg.norm(x-set1[set1_cur][0:2]),0.0, rtol=rtol, atol=atol):
                set1_on_lambda[set1_cur] = True
            if set1_on_lambda[set1_cur]:
                break
        #checking if points are zero
        if np.isclose(np.linalg.norm(set1[set1_cur][2:]),0.0, rtol=rtol, atol=atol):
            set1_on_zero[set1_cur] = True
    return np.array([set1_on_lambda, set1_on_zero])
    
#helper function for the next function
#finds minimum qualifying distance between points of set1 and points of set2
def find_min_dist(set1, set2):
    #if we're at the base level, then we have either a left or right side and simply find our
    #minimum qualifying distance
    distances = np.zeros((set1.shape[0], set2.shape[0]))
    set1_on_lambda = np.array([0]*set1.shape[0])
    set2_on_lambda = np.array([0]*set2.shape[0])
    set1_on_zero = np.array([0]*set1.shape[0])
    set2_on_zero = np.array([0]*set2.shape[0])                
    for set1_cur in range(0, set1.shape[0]):
        for set2_cur in range(0, set2.shape[0]):
            distances[set1_cur][set2_cur] = np.linalg.norm(set1[set1_cur]-set2[set2_cur])
    #getting minimum distance
    dist_min = np.nanmin(distances)
    min_val_loc = np.where(distances == dist_min)
    return [dist_min, set1[min_val_loc[0]], set2[min_val_loc[1]]]

#According to the proof, we're done if we calculate
#the intersection points of L and L_phi
#the method here is recursive
#WARNING: sort set1 and set2 before feeding to this method
def find_intersection_point(set1, set2):
    #finding the middle x value among all available points
    master_list = np.zeros((set1.shape[0]+set2.shape[0],4))
    set1_counter = 0
    set2_counter = 0
    master_counter = 0
    #shuffling together our two sorted arrays
    while(set1_counter < set1.shape[0] or set2_counter < set2.shape[0]):
        if set1_counter == set1.shape[0]:
            master_list[master_counter] = set2[set2_counter]
            master_counter += 1
            set2_counter += 1
            continue
        if set2_counter == set2.shape[0]:
            master_list[master_counter] = set1[set1_counter]
            master_counter += 1
            set1_counter += 1
            continue
        if set1[set1_counter][0] <= set2[set2_counter][0]:
            master_list[master_counter] = set1[set1_counter]
            set1_counter += 1
        else:
            master_list[master_counter] = set2[set2_counter]
            set2_counter += 1
        master_counter += 1
        
    #the coordinate value of x that partitions our sets
    partition_x = master_list[master_list.shape[0]/2][0]
    #getting the index values that partition our sets
    set1_partition = 0
    set2_partition = 0
    for cur in range(0, set1.shape[0]):
        if all(x > partition_x for x in [set1[cur][0], set2[cur][0]]):
            break
        else:
            if set1[cur][0] < partition_x:
                set1_partition += 1
            if set2[cur][0] < partition_x:
                set2_partition += 1
    #recursive part: if our partitions are not trivial, pass them into find_intersection_point
    #gets the minimum distance points on the left and right sides
    if set1_partition > 0 or set2_partition > 0:
        results = []
        #left side
        results.append(find_intersection_point(set1[0:set1_partition],set2[0:set2_partition]))
        #right side
        results.append(find_intersection_point(set1[set1_partition:], set2[set2_partition:]))
        #the two kinds of crossover possible
        results.append(find_min_dist(set1[0:set1_partition], set2[set2_partition:]))
        results.append(find_min_dist(set1[set1_partition:], set2[0:set2_partition]))
        #picking the minimum distance
        return min(results, key=lambda x: x[0])
        
    else:
        #if we're at the base level, then we have either a left or right side and simply find our
        #minimum qualifying distance
        return find_min_dist(set1, set2)



#Solving the L and L_phi approach
print("Finding intersection point between L and L_phi points...")
#####################################
#I'm working here and in the find_intersection_point function
#Reference: https://softwareengineering.stackexchange.com/questions/306063/how-to-generalize-the-planar-case-of-the-closest-pair-problem-to-d-dimensions
#####################################

#Sorting before we pass to find_intersection_point
L = L[L[:,0].argsort()]
L_phi = L_phi[L_phi[:,0].argsort()]
#this checks all the values of L to see if they are valid
#print(L)
lambda_zero = np.invert(find_lambda_or_zero(L))
#print(lambda_zero)
lambda_zero = np.logical_and(lambda_zero[0], lambda_zero[1])
#print(lambda_zero)
L = L[lambda_zero]
L_phi = L_phi[lambda_zero]

intersection_pt = find_intersection_point(L, L_phi)
print(intersection_pt)

L_intersection = np.array([np.array(intersection_pt[1][0][0:2]), np.array(intersection_pt[1][0][2:])])
print(L_intersection)
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
