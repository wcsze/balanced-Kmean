from cgitb import small
from cmath import inf
from ctypes.wintypes import SMALL_RECT
from operator import index
from cv2 import sqrt

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statistics
import math
from pyrsistent import b
from scipy.spatial import ConvexHull, convex_hull_plot_2d  
import geopy.distance
from shapely.geometry import Polygon

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
    - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
    - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
    The bearing in degrees
    :Returns Type:
    float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing) 
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def translatepoints(lat1,lon1,bearing,distancem):
    # given: lat1, lon1, b = bearing in degrees, d = distance in kilometers
    origin = geopy.Point(lat1, lon1)
    destination = geopy.distance.distance(meters=distancem).destination(point=origin, bearing=bearing)

    lat2, lon2 = destination.latitude, destination.longitude
    return lon2, lat2

def form_stretch_polygons(subregion_count,region_data,distance):
    polygons=[]
    for i in range(subregion_count):
        poly_vertices=[]
        subregion_data=region_data.loc[region_data['Subregion']==i]
        if(len(subregion_data)==0):
            continue
        points=pd.DataFrame(subregion_data,columns=['Long','Lat'])
        points=np.array(points)
        hull=ConvexHull(points)
        cx = np.mean(hull.points[hull.vertices,0])
        cy = np.mean(hull.points[hull.vertices,1])
        for vertice in hull.vertices:
            # bear from cent of polygon
            bearing = calculate_initial_compass_bearing((cy, cx), (points[vertice, 1], points[vertice, 0]))
            point=translatepoints(points[vertice, 1],points[vertice, 0],bearing,distance)
            points[vertice]=point ## replace vertices with new vertices
            poly_vertices.append(point)
        p=Polygon(poly_vertices)
        polygons.append(p)
        # for simplex in hull.simplices:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'g')

    return polygons

def get_total(labels,data):
    total_list=[]
    pre=labels[labels.index[0]]
    total=0
    for n in range(len(labels)):
        curr=labels[labels.index[n]]
        if(curr==pre):
            total+=data.iloc[labels.index[n]]['Traffic']
        else:
            total_list.append(total)
            total=data.iloc[labels.index[n]]['Traffic']
            pre=curr
        if(n==len(labels)-1):
                total_list.append(total)
    return total_list


def get_below_avg(mean,total_list):
    below_avg=[]
    for i in range(len(total_list)):
        if(total_list[i]<mean):
            below_avg.append([i,total_list[i]])
    return below_avg

def get_connectivity(polygons,below_avg):
    connect=[[]for i in range(len(below_avg))]
    for i in range(len(below_avg)):
        for j in range(len(below_avg)):
            if(polygons[below_avg[i][0]].intersects(polygons[below_avg[j][0]]) and i!=j):
                connect[i].append([below_avg[j][0],below_avg[j][1]])
    return connect 


def find_index(value,arr):
    for i in range(len(arr)):
        if(arr[i][1]==value):
            return [int(arr[i][0]),i] #[region_index,index]


def rearrange_label(region_data,cluster_count_before,cluster_count_after):
    for i in range(cluster_count_after):
        size=len(region_data.loc[region_data['Subregion']==i])
        if(size==0):
            while(True):
                for j in range(i+1,cluster_count_before):
                    region_data.loc[region_data['Subregion']==j,'Subregion']-=1
                if(len(region_data.loc[region_data['Subregion']==i])>0):
                    break
    
        

def merge_cluster(below_avg,connect,mean,polygons,region_data,total_list,region): ## problem here
    for i in range(len(below_avg)):
        below_avg[i].append(connect[i])  #combine connectivity and below_avg into 1 list
    below_avg=sorted(below_avg,key=lambda x: x[1])
    below_avg=np.array(below_avg,dtype=object)
    for i in range(len(below_avg)):
        flag=0
        to_merge=below_avg[i]
        if(len(below_avg[i][2])==0):
            continue
        below_avg[i][2]=np.array(below_avg[i][2])
        below_avg[i][2]=below_avg[i][2][below_avg[i][2][:, 1].argsort()] ##sort the connection from smallest to largest
        for j in range(len(below_avg[i][2])):
            if(abs((to_merge[1]+below_avg[i][2][j][1])-mean)<abs(to_merge[1]-mean)): ##merge only if we combine make it closer to mean
                region_data.loc[region_data['Subregion']==int(below_avg[i][2][j][0]),'Subregion']=to_merge[0]
                rearrange_label(region_data,len(polygons),len(polygons)-1)
                flag=1
                break
        if flag==1:
            break

def remove_small_cluster(region_data,subregion_count):
    cluster_count=subregion_count
    while(True):
        for i in range(cluster_count):
            cluster=region_data.loc[region_data['Subregion']==i]
            f=0
            if(len(cluster)<3):
                f=1
                indexes=region_data[region_data['Subregion']==i].index
                points=region_data.drop(indexes,axis=0)
                p1=[cluster['Long'].values[0],cluster['Lat'].values[0]]  #p1 is any point in the small cluster
                points['distance']=((points['Long']-p1[0])**2+(points['Lat']-p1[1])**2)**0.5 ##calculate distance between points and p1 
                shortest_dist=min(points['distance'])
                cluster_to_join=points.loc[points['distance']==shortest_dist,'Subregion'] ##find the closest point
                region_data.loc[region_data['Subregion']==i,'Subregion']=cluster_to_join.values[0] ##join them
                rearrange_label(region_data,cluster_count,cluster_count-1) ## rearrange the label so no missing label (ex:0,2,3)
                cluster_count-=1
                break 
        if(f==0):
            break


if __name__=="__main__":    
    raw_data=pd.read_csv(r"csvfilename") ## read the full csv( get from balanced_kmeans.py)
    raw_data_np=np.array(raw_data)
    raw_data_np=np.delete(raw_data_np,0,1)
    region_count=20 
    final=pd.DataFrame()
    for region in range(region_count):
        region_csv="region"+str(region)+"_total_new.csv"
        region_data=raw_data.loc[raw_data['Region']==region]
        subregion_count=max(region_data['Subregion'])+1   ##subregion count before merging is equal to the last index+1 
        remove_small_cluster(region_data,subregion_count)
        points=pd.DataFrame(region_data,columns=['Long','Lat']).to_numpy()
        region_data=region_data.reset_index() ##must reset the index before taking into get_total() , only need to do once
        region_data=region_data.drop(['Unnamed: 0','index'],axis=1) #delete unnecessary column
        label=pd.Series(region_data['Subregion'])
        label=label.sort_values(ascending=True)
        total_list=get_total(label,region_data)
        mean=np.mean(total_list)
            # for p in polygons:
            #     plt.plot(*p.exterior.xy)
            # plt.show()
        while(True):
            polygons=form_stretch_polygons(subregion_count,region_data,5000)
            below_avg=get_below_avg(mean,total_list)
            connect=get_connectivity(polygons,below_avg)
            # plt.scatter(points[:,0],points[:,1])
            # for p in polygons:
            #     plt.plot(*p.exterior.xy)
            # plt.show()
            merge_cluster(below_avg,connect,mean,polygons,region_data,total_list,region)
            label=pd.Series(region_data['Subregion'])
            label=label.sort_values(ascending=True)
            total_list_new=get_total(label,region_data)
            difference=set(total_list).difference(set(total_list_new)) # find if there is any difference in the total (if no difference means it is the best result then it stops looping)
            total_list=total_list_new
            if(len(difference)==0):
                    break
        final=pd.concat([final,region_data])
        score=statistics.stdev(total_list)/statistics.mean(total_list)
        print(score)
        total_list=pd.DataFrame(total_list)
        total_list.columns=['Total Traffic']
        total_list.to_csv(region_csv)
    final.to_csv('final_new.csv')