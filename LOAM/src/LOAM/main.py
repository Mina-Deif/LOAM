#!/usr/bin/env python

import copy
import numpy as np
import open3d as o3d
import airsim
import rospy
from rospy.numpy_msg import numpy_msg
import ros_numpy
from sensor_msgs.msg import PointCloud2
from rospy_tutorials.msg import Floats
import matplotlib.pyplot as plt

# from loader_vlp16 import LoaderVLP16
from loader_kitti import LoaderKITTI
from mapping import Mapper
from odometry_estimator import OdometryEstimator


def find_transformation(source, target, trans_init):
    threshold = 0.2
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.registration.registration_icp(source, target, threshold, trans_init,
    o3d.registration.TransformationEstimationPointToPlane()).transformation
    return transformation


def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd



def main():

    rospy.init_node("LOAM_node")
    rospy.loginfo("Initializing LOAM node.")
    r = rospy.Rate(10) # Node running rate
    
    # Input and output topics topic
    point_cloud_topic = rospy.get_param("/LOAM/pointcloud_topic", "airsim/VelodynePoints")########################
    map3d_topic   = rospy.get_param("/LOAM/map_topic", "LOAM/loam_map3d")############################3
    pose_topic  = rospy.get_param("/LOAM/pose_topic", "LOAM/loam_odometry") 	###########################

    map3d_pub      = rospy.Publisher(map3d_topic, numpy_msg(Floats), queue_size= 10)######################
    pose_pub     = rospy.Publisher(pose_topic, numpy_msg(Floats), queue_size= 10)###########################


    while not rospy.is_shutdown():
     rospy.loginfo("Waiting")
     Lidar_pointcloud= rospy.wait_for_message(point_cloud_topic,PointCloud2) # getting point cloud input from ros bridge (as pc2)
     rospy.loginfo("Recivied")
     Lidar_pcd_list = ros_numpy.point_cloud2.pointcloud2_to_xyz_array( Lidar_pointcloud)*100 #tranforming pc2 to xyz list
     plt.hist(Lidar_pcd_list[:,0])
     plt.show()
     print(type(Lidar_pcd_list))
     print(Lidar_pcd_list.shape)
	
     # folder = '../../alignment/numpy/'
     #folder = '/home/anastasiya/data/data_odometry_velodyne.zip/'
     loader = LoaderKITTI(Lidar_pcd_list, '00')
######################################################################################################################


     odometry = OdometryEstimator()
     global_transform = np.eye(4)
     pcds = []
     mapper = Mapper()
     for i in range(loader.length()):
         if i >= 50:
             pcd = loader.get_item(i)
             T, sharp_points, flat_points = odometry.append_pcd(pcd)
             aligned_pcds = mapper.append_undistorted(pcd[0], T, sharp_points, flat_points, vis=(i % 1 == 0))#######################
             aligned_pcds_to_publish = np.array(aligned_pcds, dtype=np.float32)##################
             map3d_pub.publish(aligned_pcds_to_publish)

     # Visual comparison with point-to-plane ICP
     pcds = []
     global_transform = np.eye(4)
     for i in range(50, 56):
         print(i)
         pcd_np_1 = get_pcd_from_numpy(loader.get_item(i)[0])
         pcd_np_2 = get_pcd_from_numpy(loader.get_item(i + 1)[0])

         T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))
         print(T)
         print(T)
         global_transform = T @ global_transform
         global_transform_to_publish=np.array(global_transform, dtype=np.float32)##################
         pose_pub.publish(global_transform_to_publish)###############################################
         pcds.append(copy.deepcopy(pcd_np_2).transform(global_transform))
         pcds.append(copy.deepcopy(pcd_np_1))

     o3d.visualization.draw_geometries(pcds)

    r.sleep()#######################################

if __name__ == '__main__':
     main()