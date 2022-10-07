import numpy as np
import sys

def line_polar_intersection(rho1, theta1, rho2, theta2):
    # Angles in radians
    #if np.sin(theta1-theta2) < 1.e-6:
    #    return
    y = (rho1*np.cos(theta2)-rho2*np.cos(theta1))/np.sin(theta1-theta2)
    x = (rho1 - y*np.sin(theta1))/np.cos(theta1)
    return (x,y)

def line_segment_intersection(seg1, seg2):

    return (x,y)
    
def line_polar_params_from_points(p1, p2):
    #theta = np.arctan((p1[0]-p2[0])/(p2[1]-p1[1]))
    theta = np.arctan2(p1[0]-p2[0], p2[1]-p1[1])
    rho   = p1[0]*np.cos(theta) + p1[1]*np.sin(theta)

    return (rho, theta)

def line_segment_approx_perpendicular(seg1, seg2):
    return True

def line_segment_approx_parallel(seg1, seg2):
    return False


def line_segment_angle(line1, line2):

    return 0

