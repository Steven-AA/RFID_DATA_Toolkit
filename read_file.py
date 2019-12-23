import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import sys
import pickle

def loss(a,b):
    x = np.abs(a-b)
    y = math.pi*2-np.abs(a-b)
    z = np.greater(x,y)
    x[z]=y[z]
    return x

def get_distance_3d(a,b):
    return np.sqrt(np.power(a[:,0]-b[0],2)+np.power(a[:,1]-b[1],2)+np.power(a[:,2]-b[2],2))

def _range(d):
    return max(d)-min(d)


class read_file:
    def __init__(self, path, tagsee_fix=False, fix_order=[0,1,3,2,5,4], pos_dic=None):
        #TODO better header
        self.data = pd.read_csv(path, header=None)
        if tagsee_fix:
            self.data = self.data[fix_order]
            self.data.columns=range(6)
            self.data[3] = (4096-self.data[3])/4096*2*math.pi
        self.data[0] = self.data[0].str.slice(-4).astype(np.int)
        self.data[1] = self.data[1].astype(np.int)
        self.data.index = map(
            lambda x: datetime.fromtimestamp(x/1e6), self.data[4])
        self.data.index = pd.to_datetime(self.data.index)
        self.ids = list(set(self.data[0]))
        self.ants = list(set(self.data[1]))
        self.allchannels = list(set(self.data[5]))
        self.allchannels.sort()
        self.time = _range(self.data[4])/1e6
        self.pos_dic = pos_dic
    
    def cac_pos_dis(self, pos_dic):
        #TODO better header
        self.data["pos"] = [list(pos_dic[_]) for _ in self.data[0]]
        self.data["dis"] = get_distance_3d(np.asarray(list(self.data["pos"])),[0,0,2.37])
        self.data.head()
    
    def get_channels(self, _id, ant):
        #TODO better header
        t = self.get_data(_id, ant)
        return list(set(t[5]))

    def _resample(self, d, rule="0.01S"):
        d = d.resample(rule).bfill()
        return d

    def get_data(self, _id=None, ant=None, c=None, p_ant=None, isresample=False, rule="0.01S"):
        #TODO better header
        t = self.data
        if _id!=None:
            t = t.loc[t[0] == _id]
        if ant!=None:
            t = t.loc[t[1] == ant]
        if c != None:
            t = t.loc[t[5] == c]
        if isresample == True:
            t = self.resample(t,rule)
        if p_ant == None:  
            return t
        mean = np.mean(t[3])
        if p_ant == 'l':
            t = t[t[3]<=mean]
        elif p_ant =='g':
            t = t[t[3]>=mean]
        return t
        
    def get_count(self, _id, ant, c=None):
        #TODO better header
        t = self.data[self.data[0] == _id]
        if c==None:
            return len(t[t[1] == ant])
        else:
            return len(t.loc[(t[1]==ant)&(t[5]==c)])

    def show_count_by_ant(self,return_data=False,show_num=True):
        #TODO better header
        n_id = len(self.ids)
        n_ant = len(self.ants)
        count = np.zeros((n_id, n_ant))
        for _, _id in enumerate(self.ids):
            for __, ant in enumerate(self.ants):
                count[_,__] = self.get_count(_id,ant)
#         for _id in self.ids:
#             for ant in self.ants:
#                 count[_id, ant] = self.get_count(_id, ant)
#         plt.figure(figsize=(20, 20))
        fig = plt.figure(figsize=(30,10))
#         fig = plt.figure()
        ax=plt.gca()
        ax.set_xticks(range(n_ant))
        ax.set_xticklabels(self.ants)
        ax.set_yticks(range(n_id))
        ax.set_yticklabels(self.ids)
        if show_num:
            for i in range(n_id):
                for j in range(n_ant):
                    text = ax.text(j, i, int(count[i, j]),
                                   ha="center", va="center", color='k',size=16)
        im = plt.imshow(count,cmap='YlGn')
#         cbar = fig.colorbar(im, ax=ax, extend='both',orientation='horizontal')
#         cbar.minorticks_on()
        if return_data==True:
            return count
    
    def show_count_by_c(self, ant, return_data=False, show_num=True):
        #TODO better header
        n_id = len(self.ids)
        n_c = len(self.allchannels)
        count = np.zeros((n_id, n_c))
        for _, _id in enumerate(self.ids):
            for __, c in enumerate(self.allchannels):
                count[_,__] = self.get_count(_id,ant,c)
#         for _id in self.ids:
#             for ant in self.ants:
#                 count[_id, ant] = self.get_count(_id, ant)
#         plt.figure(figsize=(20, 20))
        fig = plt.figure(figsize=(30,10))
#         fig = plt.figure()
        ax=plt.gca()
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(self.allchannels)
        ax.set_yticks(range(n_id))
        ax.set_yticklabels(self.ids)
        if show_num:
            for i in range(n_id):
                for j in range(n_c):
                    text = ax.text(j, i, int(count[i, j]),
                                   ha="center", va="center", color='k',size=16)
        im = plt.imshow(count,cmap='YlGn')
#         cbar = fig.colorbar(im, ax=ax, extend='both',orientation='horizontal')
#         cbar.minorticks_on()
        if return_data==True:
            return count

    def plt_RSSI(self, _id, ant):
        #TODO better header
        plt.plot(self.get_data(_id, ant)[2])

    def plt_phase(self, _id, ant):
        #TODO better header
        plt.plot(self.get_data(_id, ant)[3])

    def plt_one(self, x, y, a, z, c, s):
        #TODO better header
        plt.figure(figsize=(20, 10))
        plt.tick_params(labelsize=23)
        if z == 3: # phase
            plt.ylim((0,math.pi*2))
        if c == 0:
            c = None
        for _ in x:
            t = self.get_data(_, y, c=c)[z][a[0]:a[1]]
            plt.plot(t, s, label=str(_), linewidth=3,markersize=15)
        plt.legend()
    
    def plt_phase_diff(_tag,ant,num,best_c):
        #TODO better header
        a = self.get_data(1,1)
        num=1000
        plt.figure(figsize=(20,10))
        x = math.pi*2-a[3][:num]
        y = (math.pi*a[5][:num]*1e6/3e8*a["dis"][:num]*4+[best_c_each_ant[_][__] for _,__ in zip(a[1][:num],a[5][:num])])%(math.pi*2)
        l = loss(x,y)

        plt.plot(list(x),'o',label='real')
        plt.plot(list(y),'-',label='cac')
        th=0.6
        for i,_,__ in zip(range(len(x)),x,l):
        #     print(i,_,__)
        #     break
            if __>th:
                plt.plot(i,_,'ro',markersize=10)
        #         count+=1
        len(l[l>th])/len(x)
        plt.legend()

    def plt_any_one(self):
        #TODO better header
        channel_options = [(str(_), _) for _ in self.allchannels]
        channel_options.insert(0, ("All", 0))

        return widgets.SelectMultiple(options=self.ids, description="tag_ID:"),\
            widgets.Dropdown(options=self.ants, description="ant_ID:"),\
            widgets.IntRangeSlider(min=0, max=5000, step=200, value=[0, 500], description="data_range:"),\
            widgets.ToggleButtons(
                    options=[('RSSI', 2), ('phase', 3), ('channel', 5)],
                    value=3,
                    description='Data:',
                    disabled=False,
                ),\
            widgets.Dropdown(options=channel_options, description='Channel:'),\
            widgets.Text(
                value="-",
                description="Style:"
            )
    def cac_dis(self):
        #TODO better header
        self.data["pos"] = [list(self.pos_dic[_]) for _ in self.data[0]]
        self.data["dis"] = get_distance_3d(np.asarray(list(self.data["pos"])),[0,0,2.37])
        self.data.head()
    def version(self):
        return "0.0.14"