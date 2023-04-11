###
# Utility functions to perform coordinates transformation with QualiSys data.

# Assuming using QualiSys format:
#     pos [x, y, z]
#     rot [r_0, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8]
#         ==>
#         [ [ r_0, r_3, r_6 ]
#         [ r_1, r_4, r_7 ]
#         [ r_2, r_5, r_8 ] ]

# With orientation:

# Top View: \      /                                   X
#            \    /                                    ^
#             \  /                                     |
#              \/                                      |
#              []  (sensor)                   Y <----- Z


###

import numpy as np
from numpy.linalg import inv
import math

fov_db = {"zed": {"horizontal": 110, "vertical": 50, "horizontal_err": 3, "vertical_err": 3, "offset_x": 30, "offset_y": 205, "offset_z": -155},
          "mmwave": {"horizontal": 120, "vertical": 30, "horizontal_err": 3, "vertical_err": 3, "offset_x": -15, "offset_y": 55, "offset_z": -55}}

class CoordinateTransform():

    def global_to_local(coordinate_pos, coordinate_rot, obj_pos, obj_rot):
        """
        Convert from global coordinate to local coordinates. 
        coordinate_pos, coordinate_rot is the local coordinate system convert to.
        obj_pos, obj_rot is the obj position.
        """
    
        R_coord = coordinate_rot.reshape((3,3)).T    
        new_pos = np.dot(R_coord.T, obj_pos.T - coordinate_pos.T)
        new_rot = np.dot(obj_rot.reshape((3,3)).T, inv(R_coord))
        new_rot = new_rot.T.reshape((1,9))
        
        return new_pos, new_rot
    

    def local_to_global(coordinate_pos, coordinate_rot, obj_pos, obj_rot):
        """
        Convert from local coordinate to global coordinates. 
        coordinate_pos, coordinate_rot is the local coordinate system convert to.
        obj_pos, obj_rot is the obj position.
        """
    
        R_coord = coordinate_rot.reshape((3,3)).T
        new_pos = np.dot(R_coord, obj_pos.T) + coordinate_pos.T
        new_rot = np.dot(R_coord, obj_rot.reshape((3,3)).T)
        new_rot = new_rot.T.reshape((1,9))
        
        return new_pos, new_rot
    

    def cartesian_to_spherical(coord_pos):
        """
        Convert cartesian coordinate (x,y,z) to spherical coordinate (r, theta, phi).
        """

        r = np.sqrt(np.sum(np.square(coord_pos)))
        theta = math.atan2(coord_pos[1], coord_pos[0])
        phi = math.atan2(np.sqrt(np.sum(np.square(coord_pos[:2]))), coord_pos[2])

        return np.array([r, theta, phi])


    def spherical_to_cartesian(coord_pos):
        """
        Convert spherical coordinate (r, theta, phi) to cartesian coordinate (x,y,z)
        """

        r, theta, phi = coord_pos[0], coord_pos[1], coord_pos[2]

        x = r*math.sin(phi)*math.cos(theta)
        y = r*math.sin(phi)*math.sin(theta)
        z = r*math.cos(phi)

        return np.array([x, y, z])



class FieldOfViewCheck():

    def validate_field_of_view(self, obj_pos, sensor_name):
        """
        Check if the object is in the sensor's field of view.
        obj_pos is the local coordinate system using spherical coordinate (r, theta, phi).
        """

        r, theta, phi = obj_pos[0], math.degrees(obj_pos[1]), math.degrees(obj_pos[2])

        hoz = fov_db[sensor_name]["horizontal"] / 2.0
        hoz_err = fov_db[sensor_name]["horizontal_err"]
        vet = 90 - (fov_db[sensor_name]["vertical"] / 2.0)
        vet_err = fov_db[sensor_name]["vertical_err"]

        conf_h = 0.0
        conf_v = 0.0

        if (math.fabs(theta) - hoz) <= 0:
            conf_h = 1.0
        elif (math.fabs(theta) - hoz) <= hoz_err:
            conf_h = (math.fabs(theta) - hoz) / float(hoz_err)
        else:
            conf_h = 0.0
            
        if phi > 90:
            phi = 180 - phi

        if phi >= vet:
            conf_v = 1.0
        elif vet - phi <= vet_err:
            conf_v = (vet - phi) / float(vet_err)
        else:
            conf_v = 0.0

        return min(conf_h,conf_v)
    

    def validate_field_of_view_raw(self, coordinate_pos, coordinate_rot, obj_pos, obj_rot, sensor_name):

        temp_pos, temp_rot = CoordinateTransform.global_to_local(coordinate_pos, coordinate_rot, obj_pos, obj_rot)

        temp_pos = np.array([temp_pos[0] + fov_db[sensor_name]["offset_x"], temp_pos[1] + fov_db[sensor_name]["offset_y"], temp_pos[2] + fov_db[sensor_name]["offset_z"]])

        temp_sphe = CoordinateTransform.cartesian_to_spherical(temp_pos)

        fov_val = FieldOfViewCheck.validate_field_of_view(temp_sphe, sensor_name)

        return fov_val
