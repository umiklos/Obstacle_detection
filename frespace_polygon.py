#!/usr/bin/env python


import rospy
import std_msgs.msg as rosmsg
import nav_msgs.msg as navmsg
import sensor_msgs.msg as senmsg
import geometry_msgs.msg as geomsg # Point object
import shapely.geometry as sg      # Point object
import visualization_msgs.msg as vismsg
import numpy as np
from shapely import affinity
from shapely.geometry import Point
from sklearn import linear_model
from scipy import stats
import tf2_ros
from tf2_geometry_msgs import PointStamped


tolerance=0.12
free_space = None



class ScanSubscriber():
    def __init__(self):
        
        
        self.angles = []
        self.scan_ranges = []
        self.masked_ranges = []
        self.time = 0
        self.masked_angles = []
        self.separators = []

    def scanCallBack(self,msg):
        global free_space,tolerance
        # define the slices ~ how many nodes (points) a polygon will have (don't worry later will be simlified with shapely simplify)
        slic = 40
        if len(self.angles) < 1: # if empty
            self.angles = np.linspace(msg.angle_min, msg.angle_max, num=len(msg.ranges))
            self.separators = np.linspace(msg.angle_min, msg.angle_max, num=slic)
        self.scan_ranges = np.asarray(msg.ranges)
        self.time = msg.header.stamp
        self.masked_ranges = self.scan_ranges[(self.scan_ranges > 0.08) & (self.scan_ranges < 18)]
        self.masked_angles = self.angles[(self.scan_ranges > 0.08) & (self.scan_ranges < 18)]
        m = []
        # separators are the end of a slices
        for separator in self.separators:
            m.append(np.argmax(self.masked_angles >= separator))
        a = self.masked_angles[m]
        r = self.masked_ranges[m]
        x = r * np.cos(a)
        y = r * np.sin(a)
        free_space = sg.Polygon(np.column_stack((x, y)))
        # here the polygon will be simplified, from 40 node down to 4-7 if the barriers are in a rectangular shape
        free_space = sg.Polygon(free_space.simplify(tolerance, preserve_topology=False))

        if type(free_space) is sg.multipolygon.MultiPolygon:
            # if more polygon, choose the closest        
            min_dist = 999
            i = 0
            for poly in free_space:
                if np.mean(self.dist_poly(poly)) < min_dist:
                    min_dist = np.mean(self.dist_poly(poly))
                    min_ind = i
                i += 1
            free_space = free_space[min_ind]
        
        if free_space is not None and free_space is not []:
            free_sp_dist = self.dist_poly(free_space) # distances of the simplyfied free space

    def dist_poly(self, polygon):
        points = np.asarray(polygon.exterior.coords)
        distances = []
        for point in points:
            # eucledian distance
            distances.append(np.linalg.norm(point[0] - point[1]))
        return np.asarray(distances)
    

def linepub(): 
    rospy.init_node("rviz_marker", anonymous=True)  
    sc=ScanSubscriber()  
    rospy.Subscriber("/scan", senmsg.LaserScan, sc.scanCallBack)
    
    pub_free = rospy.Publisher("free_space", vismsg.Marker, queue_size=1)
    
    
    rate=rospy.Rate(25)
    


    # mark_f - free space marker
    mark_f = vismsg.Marker()
    mark_f.header.frame_id = "/laser"
    mark_f.type = mark_f.LINE_STRIP
    mark_f.action = mark_f.ADD
    mark_f.scale.x = 0.2
    mark_f.color.r = 0.1
    mark_f.color.g = 0.4
    mark_f.color.b = 0.9
    mark_f.color.a = 0.9 # 90% visibility
    mark_f.pose.orientation.x = mark_f.pose.orientation.y = mark_f.pose.orientation.z = 0.0
    mark_f.pose.orientation.w = 1.0
    mark_f.pose.position.x = mark_f.pose.position.y = mark_f.pose.position.z = 0.0
    
    
    

    while not rospy.is_shutdown():
        if free_space is not None:
            # marker line points
            mark_f.points = []
            pl1 = geomsg.Point(); pl1.x = -0.2; pl1.y = -1.0; pl1.z = 0.0  # x an y mismatch?
            pl2 = geomsg.Point(); pl2.x = -0.2; pl2.y = 0.6; pl2.z = 0.0
            mark_f.points.append(pl1)
            mark_f.points.append(pl2)
            try:
                if type(free_space) is sg.polygon.Polygon:                    
                    for l in free_space.exterior.coords[1:]: # the last point is the same as the first
                        p = geomsg.Point(); p.x = l[0]; p.y = l[1]; p.z = 0.0
                        mark_f.points.append(p)
            except:
                rospy.logwarn("exception")
            mark_f.points.append(pl1)
                    
            
            # Publish the Markers
            pub_free.publish(mark_f)
            
            
        rate.sleep()
    
if __name__ == '__main__':
    try:
        linepub()
    except rospy.ROSInterruptException:
        pass