import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import concurrent.futures
import copy

##########
#Overview#
##########

#This is an implementation of some concepts used in Vaughan's Theorem, an earlier similar proof.
#Particularly, Vaughan embeds all pairs of points on a curve into 3D space by considering the line segment connecting pairs. The coordinates (x,y,z) of the 3D point corresponding to a pair of points is determined as follows: (x,y) is the midpoint of the line segment between the two points, and (z) is the length of the line segment.

#There is a few reasons Vaughan found this useful
#Firstly, he knew that if this 3D surface intersected itself, that would mean two sets of points would have line segements sharing the same midpoint and the same lengths.That's equivalent to finding a rectangle!
#He knew the sets of points had the topology of a mobius strip, and a similar topology, the real projective plane, necessarily intersects itself when embedded in 3-space
#So our embedded figure (what we graph here) is a mobius strip topology, and it "sits on top of" the original curve, which lies in the xy plane. Then one "fills in" the hole within the curve on the xy plane, and the resulting object is topologically equivalent to the projective plane, which must necessarily intersect itself whenever it's embedded in 3D space. And since the intersection isn't on our boring xy plane, the intersection was caused by pairs of points!

#our precision when finding the self-intersections of our embedded surface
rtol = 1e-5
atol = 8e-2

#####################
#Data initialization#
#####################

def curve_data():
    curve = np.array([np.array([np.real((np.sin(2.0*i)+2.0)*np.exp(i*1j)),np.imag((np.sin(2.0*i)+2.0)*np.exp(i*1j))]) for i in np.arange(-0.0,2.0*pi, curve_intervals)])
    return curve

print("Hello!")
print("We're going to embed the pairs of points on a curve into 3D space")

#Jordan curve data
print("Preparing Jordan curve data...")
data = None
try:
    data = np.loadtxt("customshape.txt")
    print("   Loaded curve from customshape.txt")
except:
    print("   Failed to load custom curve; using mathematically generated curve")
    data = curve_data()

################################
#Embedding the surface into R^3#
################################
    
print("Embedding pairs of points in 3D space...")
embedded_pts = []
for pt1 in data:
    for pt2 in data:
        embedded_pts.append(np.array([(pt1[0]+pt2[0])/2.0, (pt1[1]+pt2[1])/2.0, np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)]))

embedded_pts = np.array(embedded_pts)

################################
#Calculating self-intersections#
################################
#this code is not correctly implemented
#i found intersections of our wireframe, which is way too general to get intersections
#of the actual surface
#it's also computationally expensive thus the multithreading

'''
print("Searching for possible self-intersections...")
print("   Checking {} points...".format(embedded_pts.shape[0]**2))
intersect_pts = []

#search function to find points that appear to be self-intersections
#we partition this out to multiprocessors by ranges of first points via 'start' and 'stop'
def find_intersection(start, stop):
    global embedded_pts
    intersect_pts = []
    #now we grab points which appear to overlap
    for pt1 in range(start, stop):
        for pt2 in range(0, embedded_pts.shape[0]):
            if pt1==pt2:
                continue
            else:
                if np.isclose(np.linalg.norm(embedded_pts[pt1]-embedded_pts[pt2]),0.0, rtol=rtol, atol=atol):
                    intersect_pts.append(embedded_pts[pt1])
    return intersect_pts

#multithreading shenanigans
params = []
for x in range(0, 8):
    params.append([x*embedded_pts.shape[0]/8, (x+1)*embedded_pts.shape[0]/8])

with concurrent.futures.ProcessPoolExecutor() as executor:
    future = [executor.submit(find_intersection, *param) for param in params]
results = [f.result() for f in future]
for result in results:
    intersect_pts += result
intersect_pts = np.array(intersect_pts)
'''
##################
#Plotting results#
##################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plotting our embedded surface
ax.plot_wireframe(embedded_pts[:,0], embedded_pts[:,1], embedded_pts[:,2])
#plotting our curve
ax.plot(data[:,0],data[:,1], [0]*len(data))

plt.show()
