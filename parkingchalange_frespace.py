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
import math
#from tf.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation as R



offset_from_block=rospy.get_param('/my_node/offset_from_block')       #### set parameter: how big tolerance we want from the block, the distance is in meter #### 
block_length=rospy.get_param('/my_node/block_length')       
block_length_tolerance=rospy.get_param('/my_node/block_length_tolerance')

free_space = None
orient_line = None
closest_node = None
search_pol=None
left_sideline=None
right_sideline=None
smaller_polygon=None
block_point=None
block_dist=None
h_cent=None
line_length=None
xblock=None
yblock=None
xcenter=None
ycenter=None
centerpoint=None
id_or=0
orient_mask=None
rot=None



class ScanSubscriber():
    def __init__(self):
        
        
        self.angles = []
        self.scan_ranges = []
        self.masked_ranges = []
        self.time = 0
        self.xy_masked = []
        self.masked_angles = []
        self.separators = []
        self.testvalue = []
        self.testorientation=[]

    def scanCallBack(self,msg):
        global free_space, orient_line, closest_node,search_pol,right_sideline,left_sideline,smaller_polygon,block_point,block_dist,h_cent,line_length,yblock,xblock,xcenter,ycenter,offset_from_block,block_length,block_length_tolerance,id_or,orient_mask,rot
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
        if free_space.is_valid:
        # here the polygon will be simplified, from 40 node down to 4-7 if the barriers are in a rectangular shape
            free_space = sg.Polygon(free_space.simplify(0.12, preserve_topology=False))

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
                #free_sp_dist = self.dist_poly(free_space) # distances of the simplyfied free space
                minx,miny,maxx,maxy=self.get_bounds(free_space)
                if maxx>1:
                    h_cent, orientation = self.find_orientation(free_space)
                    block_dist = self.find_block_distance(h_cent, orientation, free_space)
                    middle_coords = np.array([[0,h_cent], [0.1, h_cent + 0.1 * orientation], [block_dist, h_cent + block_dist *orientation]])
                    orient_line = sg.Polygon(list(middle_coords))
                    closest_node = block_dist #min(free_sp_dist)
                    orient_lineLs=sg.LineString(middle_coords)

                    
                    self.testvalue.append(block_dist)
                    id=len(self.testvalue)
                    block_distances=np.asarray(self.testvalue)
                    
                    block_diff=np.diff(block_distances)

                    mask=block_diff>-1                      
                    
                    if id>1:
                        if mask[id-2]==True or  (mask[id-3]==False and mask[id-2]==False) :
                            
                            if block_dist>3.5:
                            
                                crosslines,middle_line=self.cross_lines(free_space,h_cent,orientation,orient_lineLs)
                                mpoints=self.get_mpoints(crosslines)
                                left_distances,right_distances=self.getSides(middle_line,mpoints)
                                left_distance=self.get_offset(left_distances)
                                right_distance=self.get_offset(right_distances)
                                left_side_line=sg.LineString(middle_line.parallel_offset(left_distance-0.5,'left'))
                                right_side_line=sg.LineString(middle_line.parallel_offset(right_distance-0.5))  
                                
                                left_side,right_side=self.get_polygon_sides(left_side_line,right_side_line)
                                search_pol=self.get_polygon(left_side,right_side)
                                endline=self.get_endline(left_side_line,right_side_line)

                                
                                xe,ye=endline.xy
                                end_line=np.column_stack((xe,ye))
                                                        
                                polygon_diff=self.get_line_diff(endline,h_cent,block_dist)
                                ofsetted_endline=endline.parallel_offset(polygon_diff+1,'right')
                                xendl,yendl=ofsetted_endline.xy
                                endline_left=np.column_stack((xendl,yendl))

                                smaller_polygon=self.get_polygon(ofsetted_endline,end_line)

                                slope_left,intercept_left=self.lineEquations(left_side)
                                slope_right,intercept_right=self.lineEquations(right_side)
                                slope_end_r,intercept_end_r=self.lineEquations(end_line)
                                slope_end_l,intercept_end_l=self.lineEquations(endline_left)
                                
                                originalpoints=self.original_points()
                                maskedpoints=self.get_maskedpoints(slope_left,slope_right,intercept_left,intercept_right,intercept_end_r,intercept_end_l,slope_end_r,slope_end_l,originalpoints)
                                
                                if len(maskedpoints)>3:
                                    block=sg.LineString(maskedpoints)
                                    block=block.simplify(0.5,preserve_topology=False)
                                    lines,block_slopes=self.get_slope(block)       
                                    vertical=self.verical_or_horizontal(block_slopes,orientation)
                                    block_lines=lines[vertical]
                                                                
                                
                                    if np.any(block_lines) and block_lines.shape[0]>0 and len(block_lines.shape) < 4 :                            
                                        line_length,block_point,xcenter,ycenter=self.block_linesCoordinate(block_lines)
                                        xblock,yblock=block_point.xy
                                                                         
                                        line_slope=self.get_block_lines_slopes(block_lines)

                                        
                                        rot=self.get_new_orientation(line_slope)
                                        
                                        
                                        if line_length<block_length+block_length_tolerance and line_length>block_length-block_length_tolerance:
                                            rel_orientation=self.get_relative_orientation(orientation,line_slope)
                                            
                                            self.testorientation.append(rel_orientation)
                                            id_or=(len(self.testorientation))
                                            relative_orientations=np.asarray(self.testorientation)
                                            orient_diff=abs(np.diff(relative_orientations))
                                            orient_mask=orient_diff<2
                                            
                                    else:
                                        rospy.logwarn("no goal")
                                    
                                    
                                    
                                        
                                    
                ########### near block#####################################            
                            elif block_dist<3.5 and block_dist>1.5:
                            
                                crosslines,poly0=self.nearblock_get_crosslines(free_space)           
                                mpoints=self.nearblock_get_intersectpoints(crosslines)
                                if len(mpoints)>2:
                                    endline,dist=self.nearblock_get_endline(mpoints,free_space,poly0,miny,maxy)
                                    x_int_points,y_int_points,poly0_cross=self.nearblock_offset_endline(endline,dist,free_space)
                                    right_sideline,left_sideline=self.nearblock_side_lines(poly0_cross,x_int_points,y_int_points)

                                    if type(left_sideline) is sg.linestring.LineString and type(right_sideline) is sg.linestring.LineString and type(endline) is sg.linestring.LineString: 

                                        left_side,right_side=self.get_polygon_sides(left_sideline,right_sideline)

                                        offseted_endline=endline.parallel_offset(1.5)
                                        xe,ye=offseted_endline.xy
                                        xl0,yl0=poly0.xy
                                        end_line=np.column_stack((xe,ye))
                                        endline_left=np.column_stack((xl0,yl0))

                                        slope_left,intercept_left=self.lineEquations(left_side)
                                        slope_right,intercept_right=self.lineEquations(right_side)
                                        slope_end_r,intercept_end_r=self.lineEquations(end_line)
                                        slope_end_l,intercept_end_l=self.lineEquations(endline_left)

                                        left_upperP_x,left_upperP_y,right_upperP_x,right_upperP_y=self.nearblock_get_polygon(right_sideline,left_sideline,offseted_endline)
                                        left_upperP_x=np.array(left_upperP_x)
                                        left_upperP_y=np.array(left_upperP_y)
                                        right_upperP_x=np.array(right_upperP_x)
                                        right_upperP_y=np.array(right_upperP_y)
                                        x_int_points=np.array(x_int_points)
                                        y_int_points=np.array(y_int_points)
                                    
                                        left_sideNB=sg.LineString([(x_int_points[1],y_int_points[1]),(left_upperP_x,left_upperP_y)])
                                        right_sideNB=sg.LineString([(right_upperP_x,right_upperP_y),(x_int_points[0],y_int_points[0])])

                                        smaller_polygon=self.get_polygon(left_sideNB,right_sideNB)

                                        originalpoints=self.original_points()
                                        maskedpoints=self.get_maskedpoints(slope_left,slope_right,intercept_left,intercept_right,intercept_end_r,intercept_end_l,slope_end_r,slope_end_l,originalpoints)
                                        
                                        
                                        

                                        if len(maskedpoints)>3:
                                            block=sg.LineString(maskedpoints)
                                            block=block.simplify(0.5,preserve_topology=False)
                                            lines,block_slopes=self.get_slope(block)               
                                            vertical=self.nearblock_verical_or_horizontal(block_slopes,slope_end_r,slope_left)
                                            block_lines=lines[vertical]
                                            
                                            
                                            
                                            if np.any(block_lines) and block_lines.shape[0]>0 and len(block_lines.shape) < 4 :
                                                line_length,block_point,xcenter,ycenter=self.block_linesCoordinate(block_lines)
                                                xblock,yblock=block_point.xy
                            
                                                
                                                line_slope=self.get_block_lines_slopes(block_lines)

                                               
                                                rot=self.get_new_orientation(line_slope)
                                                                                         
                                                if line_length<block_length+block_length_tolerance and line_length>block_length-block_length_tolerance:
                                                    rel_orientation=self.get_relative_orientation(orientation,line_slope)
                                                    self.testorientation.append(rel_orientation)
                                                                                        
                                                    id_or=(len(self.testorientation))
                                                    relative_orientations=np.asarray(self.testorientation)
                                                    orient_diff=abs(np.diff(relative_orientations))

                                                    orient_mask=orient_diff<2
                                                    
                                            else:
                                                rospy.logwarn("no goal")
                                            
                                        

    def find_orientation(self, p):
        intersect_x = np.arange(0.1, 3, 0.4)
        # find the beginning and the end of the polygon, so the edges where a horizontal line intersects
        min_y = min(np.asarray(p.exterior.coords)[:,1])
        max_y = max(np.asarray(p.exterior.coords)[:,1])
        middle = []
        for x in intersect_x:
            l = sg.LineString([(x,min_y),(x,max_y)])            
            pp = l.intersection(p)
            if type(pp) is sg.linestring.LineString:
                x, y = pp.xy
                x_c = x[0]
                y_c = (y[0] + y[1])/ 2
                middle.append([x_c, y_c])
                
            elif type(pp) is sg.multilinestring.MultiLineString:
                x,y=pp[0].xy
                x1,y1=pp[1].xy
                x_c1=x1[1]
                y_c1=(y1[1]+y[0])/2
                middle.append([x_c1, y_c1])
               
        
        middle = np.asarray(middle)
        
        hist_num, hist_val = np.unique(np.diff(middle[:, 1]),return_counts=True)
        most_common = hist_num[np.argmax(hist_val)]
        start_y = middle[0, 1]
        middle[:, 1] = np.arange(0, len(middle[:, 1])) * most_common + start_y
        horizontal_center, orientation = np.polynomial.polynomial.polyfit(middle[:, 0], middle[:, 1], 1)
        return horizontal_center, orientation
        
            

    def find_block_distance(self, h_cent, orientation, p):
        l = sg.LineString([(0, h_cent), (20, h_cent + 20 * orientation)])
        pp = l.intersection(p)
        if type(pp) is sg.linestring.LineString:
            x, y = pp.xy
            # two intersection: at 0,0 and a the end
            if np.shape(x)[0] == 2:
                return x[1]
        else:
            return -1


    def dist_poly(self, polygon):
        points = np.asarray(polygon.exterior.coords)
        distances = []
        for point in points:
            # eucledian distance
            distances.append(np.linalg.norm(point[0] - point[1]))
        return np.asarray(distances)

    def cross_lines(self,free_space,h_cent,orientation,orient_lineLs):
        
        xa,ya=free_space.exterior.xy
        line_lenght=orient_lineLs.length+1.5
        l=sg.LineString([(0, h_cent), (line_lenght, h_cent + line_lenght * orientation)])
        crossline_0=affinity.rotate(l,90)
        poly0=sg.LineString([(xa[0],ya[0]),(xa[1],ya[1])])
        distance=poly0.distance(crossline_0)
        crossline_origin=crossline_0.parallel_offset(distance,'left')
        crossline_origin=sg.LineString(crossline_origin)
        iterator=np.arange(0.2,block_dist-0.4,0.2)

        crosslines=[]
        points=[]
        for i in range(len(iterator)):
            lines=crossline_origin.parallel_offset(iterator[i]).intersection(free_space)
            crosslines.append(lines)

        for i in range(len(crosslines)):
            if  type (crosslines[i]) is sg.linestring.LineString:
                x1,y1=crosslines[i].xy
                points.append([x1,y1])
            else:
                type(crosslines[i]) is sg.multilinestring.MultiLineString
                if len(crosslines[i])==2:
                    x1,y1=crosslines[i][0].xy
                    points.append([x1,y1])
                elif len(crosslines[i])>0:
                    x1,y1=crosslines[i][1].xy
                    points.append([x1,y1])      
                        
        points=np.array(points)   
        
        return points,l

    def get_offset(self,distances):
        rdr=np.round(distances,2)
        hist_num_r,hist_val_r=np.unique(rdr,return_counts=True)
        distance=hist_num_r[np.argmax(hist_val_r)]
        return distance
    

    def get_mpoints(self,points):

        xek=[]
        yok=[]
        for i in range(len(points)):
            xa=points[i][0]
            ya=points[i][1]
            xek.append(xa)
            yok.append(ya)
        xek=np.array(xek)
        xek=np.concatenate(xek)

        yok=np.array(yok)
        yok=np.concatenate(yok)

        mpoints=np.column_stack((xek,yok))
        return mpoints
    

    

    def getSides(self,ransac_line,mpoints):
        mx,my=ransac_line.xy
        sl,int,_,_,_=stats.linregress(mx,my)
        ymid=(mpoints[:,0]*sl)+int
        mask_left=ymid<mpoints[:,1]
        mask_right=ymid>mpoints[:,1]

        mpoints_left=mpoints[mask_left]
        mpoints_right=mpoints[mask_right]

        left_distances=[]
        for i in range(len(mpoints_left)):
            a=ransac_line.distance(Point(mpoints_left[i]))
            left_distances.append(a)

        right_distances=[]
        for i in range(len(mpoints_right)):
            b=ransac_line.distance(Point(mpoints_right[i]))
            right_distances.append(b)
        return left_distances,right_distances

    
    

    def original_points(self):
        a=self.masked_angles
        r=self.masked_ranges
        x = r * np.cos(a)
        y = r * np.sin(a)
        org=np.column_stack((x,y))
        return org

    def lineEquations(self,line):
        slope, intercept, r_value, p_value, std_err = stats.linregress(line[:,0],line[:,1])
        return slope,intercept
   

    def get_line_diff(self,endline,h_cent,block_dist):
        x,y=np.asarray(endline.centroid.xy)
        line=sg.LineString([(0,h_cent),(x,y)])
        length=line.length
        diff=length-block_dist
        return diff

    def get_slope(self,block):
        
        xb,yb=block.xy
        xb=xb[::-1]
        yb=yb[::-1]
        sl=[]
        if (len(xb))>=3:
            for i in (range(1,len(xb))):
                slo,_,_,_,_=stats.linregress([xb[i-1],(xb[i-1]+xb[i])/2,xb[i]],[yb[i-1],(yb[i-1]+yb[i])/2,yb[i]])
                sl.append(slo)
        else:
            slo,_,_,_,_=stats.linregress([xb[0],(xb[0]+xb[1])/2,xb[1]],[yb[0],(yb[0]+yb[1])/2,yb[1]])
            sl.append(slo)

        lines=np.zeros((len(xb)-1,2,2))
        for i in range(1,len(lines)+1):
            lines[i-1,0]=[xb[i-1],yb[i-1]]
            lines[i-1,1]=[xb[i],yb[i]]
        return lines,sl
        
            

    def verical_or_horizontal(self,block_slopes,orientation):
        
        block_slopes=np.array(block_slopes) 
        
        if orientation>0:
            fugg=np.where(block_slopes<0)
        elif orientation<0:
            fugg=np.where(block_slopes>0)
        return fugg


    def nearblock_verical_or_horizontal(self,sl,sl_end,sl_side):
        bb=[]
        aa=[]
        for i in range(len(sl)):
            a=abs(sl_end-sl[i])
            b=abs(sl_side-sl[i])
            aa.append(a)
            bb.append(b)    

        fugg=[]
        for j in range (len(aa)):
            if aa[j]<bb[j]:
                fugg.append(j)
                return fugg
                

    def block_linesCoordinate(self,block_lines):
        
        xf,indx=np.unique(block_lines[:,:,0],return_index=True)
        yf,indy=np.unique(block_lines[:,:,1],return_index=True)
        
        xj=xf[np.argsort(indx)]
        yj=yf[np.argsort(indy)]
        coords=np.column_stack((xj,yj))
        if len(coords)>1:
            lineb=sg.LineString((coords))
            x_centerpoint,y_centerpoint=lineb.centroid.xy
            length=lineb.length
            return length,lineb,x_centerpoint,y_centerpoint

            

    def nearblock_get_crosslines(self,free_space):
        xd,yd=free_space.exterior.xy
        poly0=sg.LineString([(xd[0],yd[0]),(xd[1],yd[1])])
        crossline0=sg.LineString([(xd[1]+1,yd[1]),(xd[1]+5,yd[1])])
        l0=poly0.length
        iterator=np.arange(0.2,l0-1,0.2)
        crosslines=[]
        crosses=[]
        for i in range(len(iterator)):
            lines=crossline0.parallel_offset(iterator[i]).intersection(free_space)
            b=crossline0.parallel_offset(iterator[i]).crosses(free_space)
            crosses.append(b)
            if crosses[i]==True:
                crosslines.append(lines)
        return crosslines,poly0

    def nearblock_get_intersectpoints(self,crosslines):
        points=[]

        for i in range(len(crosslines)):
            if  type (crosslines[i]) is sg.linestring.LineString:
                x1,y1=crosslines[i].xy
                points.append([x1,y1])
            else:
                type(crosslines[i]) is sg.multilinestring.MultiLineString
                x1,y1=crosslines[i][1].xy
                points.append([x1,y1])

        mpoints=[]
        for  i in range(len(points)):
            xa,ya=points[i][0][0],points[i][1][0]
            mpoints.append([xa,ya])
        mpoints=np.array(mpoints)
        return mpoints

    def nearblock_get_endline(self,mpoints,free_space,poly0,miny,maxy):
        
        hist_valx, hist_num = np.unique(np.diff(mpoints[:, 0]),return_counts=True)
        most_commonx = hist_valx[np.argmax(hist_num)]
        slopecommon=-0.2/most_commonx

        rc=np.round(mpoints[:,0],2)
        most_common_val,ind=np.unique(rc,return_counts=True)

        most_common_abs_val=most_common_val[np.argmax(ind)]
        interceptbl=mpoints[0,1]-(slopecommon*most_common_abs_val)

        yend=np.arange(miny-1,maxy+1,0.2)
        x_end_l=(-interceptbl+yend)/slopecommon 

        endline=sg.LineString([(x_end_l[0],yend[0]),(x_end_l[-1],yend[-1])])
        dist=poly0.distance(endline)
        return endline,dist

    def nearblock_offset_endline(self,endline,dist,free_space):
        poly0_cross=endline.parallel_offset(dist,'left')
        polmetszes=poly0_cross.intersection(free_space)
        if type(polmetszes) is sg.linestring.LineString:
            xpm,ypm=polmetszes.xy
            ypm[0]=ypm[0]+0.4
            ypm[1]=ypm[1]-0.4
        else:
            type(polmetszes) is sg.multilinestring.MultiLineString
            xpm,ypm=polmetszes[1].xy
            ypm[0]=ypm[0]+0.4
            ypm[1]=ypm[1]-0.4
        return xpm,ypm,poly0_cross

    def nearblock_side_lines(self,poly0_cross,xpm,ypm):
        right_sideline=affinity.rotate(poly0_cross,270,(xpm[0],ypm[0]))
        left_sideline=affinity.rotate(poly0_cross,90,(xpm[1],ypm[1]))
        return right_sideline,left_sideline

    def nearblock_get_polygon(self,right_sideline,left_sideline,ofsetted_endline):
        left_upperP_x,left_upperP_y=left_sideline.intersection(ofsetted_endline).xy
        right_upperP_x,right_upperP_y=right_sideline.intersection(ofsetted_endline).xy
        return left_upperP_x,left_upperP_y,right_upperP_x,right_upperP_y

    def get_bounds(self,free_space):
        minx,miny,maxx,maxy=free_space.bounds
        return minx,miny,maxx,maxy

    def get_polygon_sides(self,left_side_line,right_side_line):
        lx,ly=self.get_xy(left_side_line)
        rsx,rsy=self.get_xy(right_side_line)
        left_side=np.column_stack((lx,ly))
        right_side=np.column_stack((rsx,rsy))
        return left_side,right_side


    def get_polygon(self,left_side,right_side):
        polygon_coords=np.concatenate((left_side,right_side))
        polygon=sg.Polygon(polygon_coords)
        return polygon

    def get_xy(self,item):
        x,y=item.xy
        return x,y

    def get_endline(self,left_side_line,right_side_line):
        lx,ly=self.get_xy(left_side_line)
        rsx,rsy=self.get_xy(right_side_line)
        endline=sg.LineString([(lx[-1],ly[-1]),(rsx[0],rsy[0])])
        return endline

    def get_maskedpoints(self,slope_left,slope_right,intercept_left,intercept_right,intercept_end_r,intercept_end_l,slope_end_r,slope_end_l,originalpoints):
        yleft=self.get_equations_y(slope_left,originalpoints,intercept_left)
        yright=self.get_equations_y(slope_right,originalpoints,intercept_right)
        
        x_end_r=self.get_equations_x(slope_end_r,originalpoints,intercept_end_r)
        x_end_l=self.get_equations_x(slope_end_l,originalpoints,intercept_end_l)

        ymask=np.logical_and(originalpoints[:,1]<yleft,originalpoints[:,1]>yright)
        xmask=np.logical_and(originalpoints[:,0]<x_end_r,originalpoints[:,0]>x_end_l)
        
        mask=np.logical_and(xmask,ymask)
        maskedpoints=originalpoints[mask]
        return maskedpoints

    def get_equations_y(self,slope,points,intercept):
        eq=(slope*points[:,0])+intercept
        return eq
    
    def get_equations_x(self,slope,points,intercept):
        eq=(-intercept+points[:,1])/slope
        return eq

    def get_block_lines_slopes(self,block_lines):
        newx1=block_lines[0,0,0]
        newx2=block_lines[0,1,0]
        newy1=block_lines[0,0,1]
        newy2=block_lines[0,1,1]
        slope,_,_,_,_= stats.linregress([newx1,newx2],[newy1,newy2])
        slope=math.atan(slope)
        #slope=math.radians(slope)
        return slope

    def get_relative_orientation(self,orientation,line_slope):
        #rel_orientation=line_slope-orientation
        rel_orientation=math.radians(180)-orientation-((math.radians(90))-line_slope)
        rel_orientation=math.degrees(rel_orientation)
        return rel_orientation

    def get_new_orientation(self,line_slope):
        if line_slope< 0:
            new_orient=line_slope+(math.pi/2)
        else:
            new_orient=line_slope-(math.pi/2) 

        rot=R.from_euler('z',new_orient)
        return rot

def linepub(): 
    rospy.init_node("rviz_marker", anonymous=True)
    arr = [] 
    arr2=[] 
    sc=ScanSubscriber()
    sc.testvalue = arr
    sc.testorientation=arr2   
    rospy.Subscriber("/scan", senmsg.LaserScan, sc.scanCallBack)
    
    pub_free = rospy.Publisher("free_space_polygon", vismsg.Marker, queue_size=1)
    pub_text = rospy.Publisher("free_space_text", vismsg.Marker, queue_size=1)
    pub_wayp = rospy.Publisher("free_waypoints", vismsg.Marker, queue_size=1)
    pub_side = rospy.Publisher("side_points", vismsg.Marker, queue_size=1)
    pub_poly = rospy.Publisher("smaller_poly", vismsg.Marker, queue_size=1)
    pub_blockp=rospy.Publisher("wall", vismsg.Marker, queue_size=1)
    pub_centerpointb=rospy.Publisher("goalpoint", vismsg.Marker, queue_size=1)
    #pub_centerpoint_map=rospy.Publisher("goalpoint/map", geomsg.PointStamped, queue_size=1)
    
    rate=rospy.Rate(50)
    #tfBuffer = tf2_ros.Buffer()
    #listener = tf2_ros.TransformListener(tfBuffer)


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
    
    # mark_t - text marker
    mark_t = vismsg.Marker()
    mark_t.header.frame_id = "/laser"
    mark_t.type = mark_t.TEXT_VIEW_FACING
    mark_t.ns = "basic_shapes"
    mark_t.color.r = mark_t.color.g = mark_t.color.b = 0.7
    mark_t.color.a = 1.0 # 100% visibility
    mark_t.scale.z = 1.0
    mark_t.pose.orientation.w = 1.0
    


    # mark_w - waypoint space marker
    mark_w = vismsg.Marker()
    mark_w.header.frame_id = "/laser"
    mark_w.type = mark_w.LINE_STRIP
    mark_w.action = mark_w.ADD
    mark_w.scale.x = 0.2
    mark_w.color.r = 0.9
    mark_w.color.g = 0.2
    mark_w.color.b = 0.1
    mark_w.color.a = 0.9 # 90% visibility
    mark_w.pose.orientation.x = mark_w.pose.orientation.y = mark_w.pose.orientation.z = 0.0
    mark_w.pose.orientation.w = 1.0
    mark_w.pose.position.x = mark_w.pose.position.y = mark_w.pose.position.z = 0.0

    
    # mark_s - sidepoint space marker
    mark_s = vismsg.Marker()
    mark_s.header.frame_id = "/laser"
    mark_s.type = mark_s.LINE_STRIP
    mark_s.action = mark_s.ADD
    mark_s.scale.x = 0.2
    mark_s.color.r = 0.1
    mark_s.color.g = 0.9
    mark_s.color.b = 0.6
    mark_s.color.a = 0.9 # 90% visibility
    mark_s.pose.orientation.x = mark_s.pose.orientation.y = mark_s.pose.orientation.z = 0.0
    mark_s.pose.orientation.w = 1.0
    mark_s.pose.position.x = mark_s.pose.position.y = mark_s.pose.position.z = 0.0

    # mark_sm - smaller polygon marker
    mark_sm = vismsg.Marker()
    mark_sm.header.frame_id = "/laser"
    mark_sm.type = mark_sm.LINE_STRIP
    mark_sm.action = mark_sm.ADD
    mark_sm.scale.x = 0.2
    mark_sm.color.r = 0.9
    mark_sm.color.g = 1
    mark_sm.color.b = 0.4
    mark_sm.color.a = 0.9 # 90% visibility
    mark_sm.pose.orientation.x = mark_sm.pose.orientation.y = mark_sm.pose.orientation.z = 0.0
    mark_sm.pose.orientation.w = 1.0
    mark_sm.pose.position.x = mark_sm.pose.position.y = mark_sm.pose.position.z = 0.0

    
    # mark_e - endpoint space marker
    mark_e = vismsg.Marker()
    mark_e.header.frame_id = "/laser"
    mark_e.type = mark_e.LINE_STRIP
    mark_e.action = mark_e.ADD
    mark_e.scale.x = 0.2
    mark_e.color.r = 0.8
    mark_e.color.g = 0.5
    mark_e.color.b = 0
    mark_e.color.a = 0.9 # 90% visibility
    mark_e.pose.orientation.x = mark_e.pose.orientation.y = mark_e.pose.orientation.z = 0.0
    mark_e.pose.orientation.w = 1.0
    mark_e.pose.position.x = mark_e.pose.position.y = mark_e.pose.position.z = 0.0


    

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
                    #mark_t.text = str(np.shape(free_space.exterior.coords)[0] - 1) + "db"
                    mark_t.text = ("%.3f") % (closest_node) 
                    for l in free_space.exterior.coords[1:]: # the last point is the same as the first
                        p = geomsg.Point(); p.x = l[0]; p.y = l[1]; p.z = 0.0
                        mark_f.points.append(p)
            except:
                rospy.logwarn("exception")
            mark_f.points.append(pl1)
            # orient_line
            mark_w.points = []
            if orient_line is not None and orient_line is not []:
                for w in orient_line.exterior.coords:
                    p = geomsg.Point(); p.x = w[0]; p.y = w[1]; p.z = 0.0
                    mark_w.points.append(p)
            
            mark_s.points=[]        
            if search_pol is not None and search_pol is not []:
                for s in search_pol.exterior.coords:
                    p = geomsg.Point(); p.x = s[0]; p.y = s[1]; p.z = 0.0
                    mark_s.points.append(p)

            mark_sm.points=[]        
            
            
            
            mark_e.points=[]
            if id_or > 1:
                if orient_mask[id_or-2]==True  or orient_mask[id_or-3]==False and orient_mask[id_or-2]==False :

                    if smaller_polygon is not None and smaller_polygon is not [] and block_dist >-1:
                        for s in smaller_polygon.exterior.coords:
                            p = geomsg.Point(); p.x = s[0]; p.y = s[1]; p.z = 0.0
                            mark_sm.points.append(p)

                    if block_point is not None and block_point is not [] and line_length<block_length+block_length_tolerance and line_length>block_length-block_length_tolerance and block_dist >-1:
                        for e in range(2):
                            p = geomsg.Point(); p.x = xblock[e]; p.y = yblock[e]; p.z = 0.0
                            mark_e.points.append(p)
                    """        
                    else:
                        rospy.logwarn("no wall found")
                    """
                    if ycenter is not None and xcenter is not None and block_point is not None and block_point is not [] and line_length<block_length+block_length_tolerance and line_length>block_length-block_length_tolerance and block_dist >-1 and rot is not None:
                        
                        
                        mark_d = vismsg.Marker()
                        mark_d.header.frame_id = "/laser"
                        mark_d.type = mark_d.SPHERE
                        mark_d.action = mark_d.ADD
                        mark_d.pose.position.x=xcenter[0]-offset_from_block
                        mark_d.pose.position.y=ycenter[0]
                        mark_d.pose.position.z=0.0
                        mark_d.pose.orientation.x=rot.as_quat()[0]
                        mark_d.pose.orientation.y=rot.as_quat()[1]
                        mark_d.pose.orientation.z=rot.as_quat()[2]
                        mark_d.pose.orientation.w=rot.as_quat()[3]
                        mark_d.scale.x = mark_d.scale.y=mark_d.scale.z=0.5
                    
                        mark_d.color.r = 0.9
                        mark_d.color.g = 1
                        mark_d.color.b = 0.4
                        mark_d.color.a = 0.9 # 90% visibility
                        
                        pub_centerpointb.publish(mark_d)
                    """    
                    else:
                        rospy.logwarn("no centerpoint")

                       
                        try:
                            target_pt = tfBuffer.transform(mark_d, "map")
                            pub_centerpoint_map.publish(target_pt)
                        except:
                            continue
                        """           
                    

            
            
            # Publish the Markers
            pub_free.publish(mark_f)
            pub_text.publish(mark_t)
            pub_wayp.publish(mark_w)
            pub_side.publish(mark_s)
            pub_poly.publish(mark_sm)
            pub_blockp.publish(mark_e)
            
            
        rate.sleep()
    
if __name__ == '__main__':
    try:
        linepub()
    except rospy.ROSInterruptException:
        pass