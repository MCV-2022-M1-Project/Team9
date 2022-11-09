import numpy as np
import cv2

class Rotation:   

    @staticmethod
    def fix_image_rotation(image):
        """
        Given an image, estimates the rotation of the paintings in it and returns the image after correcting the rotation. 

        """
        #estimate rotation angle 

        #rotate image -angle degrees to correct the rotation
        
        ###TODO (PLACEHOLDER CODE)
        angle = 0
        rotated_image = image
        ###
        return rotated_image, angle
    
    @staticmethod
    def rotate_coordinates(top_left, top_right, bottom_right, bottom_left, angle):
        """Given 4 coordinates and an angle, returns the same coordinates rotated by angle degrees
            top_left, top_right, bottom_right, bottom_left: Coordinates in the format (x,y)
            angle: angle in degrees between 0 and 180
        """
        ###TODO 
        rotated_points = Rotation.rotate([top_left, top_right,bottom_right,bottom_left])
        top_left_rotated = rotated_points[0]
        top_right_rotated = rotated_points[1]
        bottom_right_rotated = rotated_points[2]
        bottom_left_rotated = rotated_points[3]
        """
        top_left_rotated = top_left
        top_right_rotated = top_right
        bottom_right_rotated = bottom_right
        bottom_left_rotated = bottom_left
        """
        return top_left_rotated, top_right_rotated, bottom_right_rotated, bottom_left_rotated
        
        ##!!!UNTESTED!!!
    def rotate(p, origin=(0, 0), degrees=0):
        """
        From https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        """
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)