from ctypes import sizeof
from multiprocessing.spawn import freeze_support
import re
from matplotlib.pyplot import scatter
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import block
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pandas.core.construction import array
from sklearn.datasets import make_blobs 

import random 
import statistics

from sklearn.cluster import KMeans
import math
import time 
import threading

    
def balanced_cluster(data,k,X,n,long_range,lat_range):
    min_sd=math.inf
    optimal_total=[]
    for i in range(n):
        total_list=[]
        init_centroid=[]
        for cluster in range(k):
            init_centroid.append([random.uniform(long_range[0],long_range[1]),random.uniform(lat_range[0],lat_range[1])])
        init_centroid=np.array(init_centroid)
        model1=KMeans(n_clusters=k,init=init_centroid,n_init=1).fit(data)
        mb1 = pd.Series(model1.labels_)
        #time1=time.perf_counter()
        mb1=mb1.sort_values(ascending=True)
        pre_k=mb1[mb1.index[0]]
        total=0
        for n in range(len(mb1)):
            curr_k=mb1[mb1.index[n]]
            if(curr_k==pre_k):
                total+=X[mb1.index[n]][3]
            else:
                total_list.append(total)
                total=X[mb1.index[n]][3]
                pre_k=curr_k
            if(n==len(mb1)-1):
                total_list.append(total)
        #time2=time.perf_counter()
        #print(time2-time1)

        sd=statistics.stdev(total_list)
        if(sd<min_sd):
            min_sd=sd 
            optimal_centroid=init_centroid
            optimal_total=total_list
            
    
    score=min_sd/statistics.mean(optimal_total)
    return optimal_centroid,score


def group_cell(raw_data):
    grouped_data=[]
    pre=raw_data[0]
    total=0
    ##group same lat long##
    for i in range(len(raw_data)):
        curr=raw_data[i]
        if(curr[0]==pre[0]):
            total+=curr[3]
        else:
            grouped_data.append([pre[0],pre[1],pre[2],total])
            pre=raw_data[i]
            total=raw_data[i][3]
        if(i==len(raw_data)-1):
            grouped_data.append([curr[0],curr[1],curr[2],total])
    return grouped_data

def get_total(labels,data):
    total_list=[]
    pre=labels[labels.index[0]]
    total=0
    for n in range(len(labels)):
        curr=labels[labels.index[n]]
        if(curr==pre):
            total+=data[labels.index[n]][3]
        else:
            total_list.append(total)
            total=data[labels.index[n]][3]
            pre=curr
        if(n==len(labels)-1):
                total_list.append(total)
    return total_list

def get_ratio(region_count,region_total):
    ratio=[]#store the ratio of traffic of each region
    for i in range(region_count):
        ratio.append(region_total[i]/sum(region_total))
    return ratio

def get_region_data(region_count,labels,grouped_data):
    region_data=[[]for i in range(region_count)] ##to store points in each region cluster
    for i in range(len(labels)):
        region_data[labels[labels.index[i]]].append(list(grouped_data[labels.index[i]]))
    return region_data

def add_label(region_data,region_count):
    for i in range(region_count):
        for e in region_data[i]:
            e.append(i)
    return region_data

def balance(region_count,region_data,max_cluster,ratio,n):
    final=pd.DataFrame()
    for i in range(region_count):
        df=np.array(region_data[i])
        data=np.delete(df,2,1)
        data=np.delete(data,2,1)
        data=np.delete(data,2,1)
        max_k=int(max_cluster*ratio[i])
        min_k=int(0.875*max_k)
        optimal_score=math.inf
        optimal_centroid=[]
        for k in range(min_k,max_k+1):
            long_range=[np.min(df[:,0]),np.max(df[:,0])]  #x axis
            lat_range=[np.min(df[:,1]),np.max(df[:,1])]  #y axis
            avg_traffic=sum(df[:,2])/k
            name="region"+str(i)
            optimal_model=balanced_cluster(data,k,df,n,long_range,lat_range)
            score=optimal_model[1]
            if(score<optimal_score):
                optimal_score=score
                optimal_centroid=optimal_model[0]
                optimal_k=k
        model=KMeans(n_clusters=optimal_k,init=optimal_centroid,n_init=1).fit(data)
        model_label = pd.Series(model.labels_)
        model_label=model_label.sort_values(ascending=True)
        total_list=get_total(model_label,df)
        for i in range(optimal_k):
            plt.scatter(df[model.labels_==i,0],df[model.labels_==i,1],s=10,cmap=plt.cm.coolwarm,label=total_list[i])
        plt.legend(title='cluster traffic',prop={'size': 6})
        img_path=name+'.pdf'
        fig = plt.gcf()
        fig.set_size_inches((8.5, 11), forward=False)
        plt.savefig(img_path)
        plt.clf()
        df=pd.DataFrame(df,columns=['Long','Lat','EnodebID','Traffic','Region'])
        df['Subregion']=pd.Series(model.labels_)
        time1=time.perf_counter()
        final=pd.concat([final,df])
        df1=total_list
        print(statistics.stdev(total_list)/statistics.mean(total_list))
        df1=pd.DataFrame(df1)
        df1.columns=['Total Traffic']
        csv_path=name+'_total.csv'
        df1.to_csv(csv_path)
    return final
if __name__=="__main__":
##Data preprocessing########################################################################
    raw_data=pd.read_csv(r"csvfilename",usecols=['Lat','Long','Traffic','EnodebID'])
    #df=pd.DataFrame.sample(df,n=int(len(df)/5))
    raw_data=raw_data.sort_values(by=['Long'],ascending=True)
    raw_data=np.array(raw_data)
    grouped_data=group_cell(raw_data)## group cell with same lat long
    grouped_data=np.array(grouped_data)
    input_data=np.delete(grouped_data,2,1)##drop last column(traffic)
    input_data=np.delete(input_data,2,1)##drop last column(enobid)
    region_count=20
    max_cluster=400
    model=KMeans(n_clusters=region_count).fit(input_data)
    labels=pd.Series(model.labels_)
    labels=labels.sort_values(ascending=True)## sort it first before bringing into get_total()
    region_total=get_total(labels,grouped_data)
    ratio=get_ratio(region_count,region_total)
    
    region_data=get_region_data(region_count,labels,grouped_data) ##to store points in each region cluster
    region_data=add_label(region_data,region_count)##add region label 
    

    for i in range(region_count):
        plt.scatter(np.array(region_data[i])[:,0],np.array(region_data[i])[:,1],s=10,label=region_total[i],cmap=plt.cm.coolwarm)
    plt.legend(title='Region Total Traffic',loc='upper center')
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    plt.savefig('region.jpg')
    plt.clf()
# # # #############################################################################   
# # # ##balance clustering 
    t1=time.perf_counter()
    final=balance(region_count,region_data,max_cluster,ratio,10000)
    final.to_csv('final.csv')
    t2=time.perf_counter()
    print(t2-t1)



        
    
    
    
    

    
    

