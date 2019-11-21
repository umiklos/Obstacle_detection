#!/usr/bin/env python

import rospy
import std_msgs.msg as rosmsg
import nav_msgs.msg as navmsg
import sensor_msgs.msg as senmsg
import numpy as np
from sklearn import linear_model, datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from numpy import inf
import matplotlib.pyplot as plt
import math
from scipy.stats import linregress
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray



class ScanSubscriber():
    def __init__(self):
        self.Cx=None
        self.Cy=None
        self.fx=None
        self.fy=None
        self.dist=None
        self.orient=None
        #rospy.init_node("listener", anonymous=True)    
        self.scan_sub = rospy.Subscriber("/scan", senmsg.LaserScan, self.scanCallBack)
        self.fxa=None
        self.fya=None


    def scanCallBack(self,msg):
        #a_start=rospy.Time.now()
        #global first_use
        self.scan_ranges = msg.ranges
        self.angles = np.linspace(msg.angle_min, msg.angle_max, num=len(self.scan_ranges))
        self.time = msg.header.stamp        

        x=self.scan_ranges * np.cos(self.angles)
        x[x== -inf]=0
        x[x== inf]=0
        x_a=x[np.nonzero(x)]
        y=self.scan_ranges * np.sin(self.angles)
        y[y== -inf]=0
        y[y== inf]=0
        y_a=y[np.nonzero(y)]
        xy=np.column_stack((x_a,y_a))
        x_mask=xy[:,0]>0
        
        self.xy_masked=xy[x_mask]

        #print(len(self.xy_masked))

        #print(len(self.scan_ranges))

      
        x_wall_r,y_wall_r=[],[]
    
        for i in range(60):
            x1_wall_r=np.cos(self.angles[i])*self.scan_ranges[i]
            y1_wall_r=np.sin(self.angles[i])*self.scan_ranges[i]
            #x1_wall_r=self.xy_masked[i,0]
            #y1_wall_r=self.xy_masked[i,1]
            x_wall_r.append(x1_wall_r)
            y_wall_r.append(y1_wall_r)
        x_wall_r=np.array(x_wall_r)
        y_wall_r=np.array(y_wall_r)

        x_wall_l,y_wall_l=[],[]
    
        for i in reversed(range(-60,0)):
            x1_wall_l=np.cos(self.angles[i])*self.scan_ranges[i]
            y1_wall_l=np.sin(self.angles[i])*self.scan_ranges[i]
            #x1_wall_l=self.xy_masked[i,0]
            #y1_wall_l=self.xy_masked[i,1]
            x_wall_l.append(x1_wall_l)
            y_wall_l.append(y1_wall_l)
        x_wall_l=np.array(x_wall_l)
        y_wall_l=np.array(y_wall_l)


        slope_R, intercept_R, _,_,_ = linregress([x_wall_r,y_wall_r])
        Y_Right_wall=slope_R*self.xy_masked[:,0]+intercept_R 

        slope_L, intercept_L,_,_,_= linregress([x_wall_l,y_wall_l])
        Y_Left_wall=slope_L*self.xy_masked[:,0]+intercept_L
        
        y_mask_R=self.xy_masked[:,1] > Y_Right_wall+0.1  
        y_mask_L=self.xy_masked[:,1] < Y_Left_wall-0.1

        y_mask=np.logical_and(y_mask_L,y_mask_R)

        self.xy_y_masked=self.xy_masked[y_mask] 
        

        """
        plt.scatter(self.xy_masked[:,0],self.xy_masked[:,1])
        
        plt.plot(self.xy_masked[:,0],Y_Left_wall,color='r')
        plt.plot(self.xy_masked[:,0],Y_Right_wall,color='g')
        
        plt.show()
        """
        
    #def clustering(self):
        if self.xy_y_masked.shape[0] > 0:
            
            scaler=StandardScaler()
            X_scaled=scaler.fit_transform(self.xy_y_masked)
            dbscan=DBSCAN(eps=0.4,min_samples=5).fit(self.xy_y_masked)
            

            clusters=dbscan.fit_predict(X_scaled)
            cl=clusters
            
            
            self.n_clusters=len(set(cl))-(1 if -1 in cl else 0)      
            merge = np.concatenate((np.reshape(cl, (-1, 1)), self.xy_y_masked), axis=1)
            #print(set(cl),self.n_clusters)

           
            

            self.ylist = []
            self.xlist = []
            for i in range(int(min(merge[:,0])), int(max(merge[:,0]+1))):
                tol = np.where(merge[:,0] == i)[0][0]
                ig = np.where(merge[:,0] == i)[0][-1]
                self.xlist.append(merge[tol:ig+1, 1:2])
                self.ylist.append(merge[tol:ig+1, 2:3])

            self.xlist=np.array(self.xlist)
            self.ylist=np.array(self.ylist)

        
    
            LX_ransac = []
            LY_ransac = []
                
            ransac = linear_model.RANSACRegressor(max_trials=50,min_samples=2)

            for j in range (0,len(self.xlist)):
                
                #print(self.xlist.shape)
                    
                #    print(self.xlist.size,len(self.xlist))
                if len(self.xlist[j])> 2:                
                    diff=self.xlist[j].max()-self.xlist[j].min()
                    d=math.floor(math.log10(diff))
                    ransac.fit(self.xlist[j],self.ylist[j])
                    inlier_mask = ransac.inlier_mask_
                    

                    line_X = np.arange(self.xlist[j].min(), self.xlist[j].max(),diff-(10**(d-2)))[:, np.newaxis]
                    LX_ransac.append(line_X)
                

                    line_y_ransac = ransac.predict(line_X)
                    LY_ransac.append(line_y_ransac)
                    
                            

            LX_ransac=np.array(LX_ransac)    
            LY_ransac=np.array(LY_ransac)   
            
            
            sample_length=np.empty([len(LX_ransac),1])   

            #print(sample_length) 
    
    
            for f in range (0,len(LX_ransac)):
                sample_length[f,0]=math.sqrt(((LX_ransac[f,0]-LX_ransac[f,1])**2)+(LY_ransac[f,0]-LY_ransac[f,1])**2)
    
            

            self.fx=np.empty([2,0], dtype=float)
            self.fy=np.empty([2,0],dtype=float)

        
            for k in range(len(sample_length)):
                if sample_length[k,0] > 3.5 and sample_length[k,0] < 3.8:
                    self.fx=LX_ransac[k,:]
                    self.fy=LY_ransac[k,:]
                

            if self.fx.size > 0:
                self.fxa=np.array(self.fx, dtype=float) 
                self.fya=np.array(self.fy, dtype=float)
                

            if self.fx.shape[1] > 0:
                self.Cx=((self.fxa.max()-self.fxa.min())/2)+self.fxa.min()
                self.Cy=((self.fya.max()-self.fya.min())/2)+self.fya.min()
                

                
                self.dist=math.sqrt((self.Cx**2)+(self.Cy**2))
                self.orient=math.tan(self.Cy/self.Cx)
                self.orient_d=math.degrees(self.orient)
    #a_end=rospy.Time.now()
    #print(a_end-a_start)   
                    
       
def linepub():  
      
    rospy.init_node("rviz_marker", anonymous=True)  
    sc=ScanSubscriber()  
    scan_sub = rospy.Subscriber("/scan", senmsg.LaserScan,sc.scanCallBack)
    #rospy.spin()
    
    
    topic = 'obstacle_line'
    publisher = rospy.Publisher(topic, MarkerArray,queue_size=2)
    topic2='obstacle_center'
    publisher2 = rospy.Publisher(topic2, Marker,queue_size=2)
    rate=rospy.Rate(2)
    #rospy.init_node('rviz_markers')

    markerArray= MarkerArray()
    marker1 = Marker()
    marker1.header.frame_id = "/laser"
    marker1.type = marker1.SPHERE
    marker1.action = marker1.ADD
    marker1.scale.x = 0.1
    marker1.scale.y = 0.1
    marker1.scale.z = 0.1
    marker1.color.a = 1.0
    marker1.color.r = 0.0
    marker1.color.b = 0.0
    marker1.color.g = 255.0
    marker1.id=0
    marker1.pose.orientation.x = 0.0
    marker1.pose.orientation.y = 0.0
    marker1.pose.orientation.z = 0.0
    marker1.pose.orientation.w = 1.0
    
    marker2 = Marker()
    marker2.header.frame_id = "/laser"
    marker2.type = marker2.SPHERE
    marker2.action = marker2.ADD
    marker2.scale.x = 0.1
    marker2.scale.y = 0.1
    marker2.scale.z = 0.1
    marker2.color.a = 1.0
    marker2.color.r = 0.0
    marker2.color.b = 0.0
    marker2.color.g = 255.0
    marker2.id=1
    marker2.pose.orientation.x = 0.0
    marker2.pose.orientation.y = 0.0
    marker2.pose.orientation.z = 0.0
    marker2.pose.orientation.w = 1.0

    marker3 = Marker()
    marker3.header.frame_id = "/laser"
    marker3.type = marker3.SPHERE
    marker3.action = marker3.ADD
    marker3.scale.x = 0.1
    marker3.scale.y = 0.1
    marker3.scale.z = 0.1
    marker3.color.a = 1.0
    marker3.color.r = 255.0
    marker3.color.b = 0.0
    marker3.color.g = 165.0
    marker3.pose.orientation.x = 0.0
    marker3.pose.orientation.y = 0.0
    marker3.pose.orientation.z = 0.0
    marker3.pose.orientation.w = 1.0
    

    while not rospy.is_shutdown():
        

        marker1.lifetime= rospy.Duration(0)
        marker2.lifetime= rospy.Duration(0)

        if not sc.fxa is None:

            marker1.pose.position.x=sc.fxa[0]
            marker1.pose.position.y=sc.fya[0]
            marker1.pose.position.z=0

            marker2.pose.position.x=sc.fxa[1]
            marker2.pose.position.y=sc.fya[1]
            marker2.pose.position.z=0

            marker3.pose.position.x=sc.Cx
            marker3.pose.position.y=sc.Cy
            marker3.pose.position.z=0
            #print(marker3)
            markerArray.markers.append(marker1)
            
            markerArray.markers.append(marker2)

            publisher.publish(markerArray)
            publisher2.publish(marker3)
        else:
            rospy.logwarn('no goal found')
            
            
            
        rate.sleep()
    
        
    
if __name__ == '__main__':
    try:
        linepub()
        #centerpub()
    except rospy.ROSInterruptException:
        pass
        
    






