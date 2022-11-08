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
            angle: angle in degrees
        """
        ###TODO 
        top_left_rotated = None
        top_right_rotated = None
        bottom_right_rotated = None
        bottom_left_rotated = None

        return top_left_rotated, top_right_rotated, bottom_right_rotated, bottom_left_rotated
        
