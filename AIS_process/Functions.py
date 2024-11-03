import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.interpolate import interp1d
from geopy.distance import geodesic



def timestamp_trans(dt):
    dt_strp=time.strptime(dt,'%Y-%m-%d %H:%M:%S')
    dt_trans=time.mktime(dt_strp)
    return dt_trans


def date_trans(dt):
    timeArray = time.localtime(dt)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

def Remove_Error(df, start_comp, end_comp):
    df = df[df["Speed (kn)"] <= 25]
    mmsi_del = []
    grouped = df.groupby(by="MMSI")
    for i,v in grouped:
        df_one = v
        df_one_1 = df_one[df_one["Speed (kn)"] > 1 ]
        prop = df_one_1.shape[0]/df_one.shape[0]*100 # the rate of low speed point / all point
        start_gap = int((timestamp_trans(str(v.iloc[0, 0])) - start_comp) / 60) # too short traj
        end_gap = int((end_comp - timestamp_trans(str(v.iloc[-1, 0]))) / 60)
        gap = start_gap + end_gap
        if prop<50:
            mmsi_del.append(i)

        elif gap > 4:
            mmsi_del.append(i)
    df = df[~df["MMSI"].isin(mmsi_del)]
    df = df[df["Speed (kn)"] >= 0.2]
    grouped = df.groupby(by="MMSI")
    less_miss = [i for i, v in grouped if v.shape[0]< 2]
    df = df[~df["MMSI"].isin(less_miss)]

    return df

def point_distance_difference(df):
    v = df.copy()
    v["Lon_next"] = v["Longitude (deg)"].shift(-1).copy()
    v["Lat_next"] = v["Latitude (deg)"].shift(-1).copy()
    a = v["Longitude (deg)"]
    b = v["Latitude (deg)"]
    c = v["Lon_next"]
    d = v["Lat_next"]
#   D = relative_distance(a,b,c,d)
    D = np.sqrt((c-a)**2+(d-b)**2)*60*1.852
    return D

def insert_func(ser, rep, kind):

    x = np.linspace(0, 1, len(ser))
    x_in = np.linspace(0, 1, rep)
    func = interp1d(x, ser, kind=kind)

    return func(x_in)


def G_insert(df_in, min_repoint, date):


    coord = df_in.iloc[:, 2:4].reset_index(drop=True)
    feats = df_in.iloc[:, 2:].reset_index(drop=True)
    # insert for breakpoint
    for i in range(min_repoint):
        distance = point_distance_difference(coord)
        over_dis = distance[distance > 0.5].index.tolist()
        if over_dis == []:
            break
        else:  # interpolate breakpoints
            split = feats.iloc[over_dis[0]:over_dis[0] + 2, :]
            rep = int(np.sqrt((split.iloc[1,0] - split.iloc[0,0])**2+(split.iloc[1,1] - split.iloc[0,1])**2)*60*1.852 / 0.3)+1
            splited = pd.DataFrame()
            splited['Longitude (deg)'] = insert_func(split['Longitude (deg)'], rep, kind='linear')
            splited['Latitude (deg)']  = insert_func(split['Latitude (deg)'], rep, kind='linear')
            splited['Speed (kn)'] = insert_func(split['Speed (kn)'], rep, kind='linear')
            splited['Heading (deg)'] = insert_func(split['Heading (deg)'], rep, kind='linear')
            splited['Length (m)'] = [np.mean(feats['Length (m)'].values)] * rep
            feats = pd.concat([feats.iloc[:over_dis[0],:],splited,feats.iloc[over_dis[0]+2:,:]]).reset_index(drop=True)

    # insert for whole

    date['MMSI'] = [df_in.values[1,1]] * min_repoint
    date['Longitude (deg)'] = insert_func(feats['Longitude (deg)'], min_repoint, kind='linear')
    date['Latitude (deg)'] = insert_func(feats['Latitude (deg)'], min_repoint, kind='linear')
    date['Speed (kn)'] = insert_func(feats['Speed (kn)'], min_repoint, kind='linear')
    date['Heading (deg)'] = insert_func(feats['Heading (deg)'], min_repoint, kind='linear')
    date['Length (m)'] = [np.mean(feats['Length (m)'].values)] * min_repoint
    out = date.copy()

    return out


def Show_Tra_Completed(df,start_comp,end_comp):
    grouped = df.groupby(by='MMSI')
    no_comp = 0
    no_comp_list =[]
    for n,(i,ii) in enumerate(grouped):
        start_gap = int((timestamp_trans(str(ii.iloc[0,0])) - start_comp)/60)
        end_gap = int((end_comp - timestamp_trans(str(ii.iloc[-1,0])))/ 60)
        if start_gap > 2 or end_gap > 2:
            no_comp = no_comp + 1
            no_comp_list.append((i,start_gap,end_gap))
    rate_comp = int((n+1 - no_comp) / (n+1)) * 0.1
    return [rate_comp, no_comp, n+1],no_comp_list


def Show_Batch(batch, map_root, map_area):
    plt.scatter(batch[:,:, 2], batch[:,:, 3], c='b', s=3)
    imp = plt.imread(map_root)
    plt.imshow(imp, extent = map_area)
    plt.show()

    return

def Show_DFson(df_son, map_root, map_area):
    plt.scatter(df_son.iloc[:, 2], df_son.iloc[:, 3],c='b', s=3)
    imp = plt.imread(map_root)
    plt.imshow(imp, extent= map_area)
    plt.show()
    return


def get_ais_small():
    '''
    get part of data from A month AIS data
    a day: 8:640944 and a week: 8:4374283 form 22 Y 09 M start
    '''
    AIS_pos_raw = '/Users/yangkaisen/MyProject/Data/20220908-20220930_final.csv'
    small_ais_save = '/Users/yangkaisen/MyProject/Data/test.csv'
    df = pd.read_csv(AIS_pos_raw, skiprows= lambda i: i>4374283)
    df = df.iloc[8:,:]
    last = df.iloc[-100:,:2]
    first = df.iloc[:10,:2]
    new = open(small_ais_save, 'w')
    df.to_csv(new,index=False)
    new.close()
    print(';')
    return

def Get_virtual_pos(ll, channel_pos):
    '''
    Get_virtual_pos for every actual_pos, and transfor SOG and COG to speed
    :param channel_pos: a ndarray with "log" and "lat" 2 columns, [N, 2]
    '''
    channel_pos = np.zeros((200, 2))
    channel_pos[:, 0] = np.random.uniform(114.099003, 114.187537, 200)
    channel_pos[:, 1] = np.random.uniform(22.265695, 22.322062, 200)

    actual_nodes = ll.shape[0]
    virtual_pos = np.zeros((actual_nodes, 2))

    for pos in range(actual_nodes):
        a = ll["Longitude (deg)"][pos]
        b = ll["Latitude (deg)"][pos]
        c = channel_pos[:, 0]
        d = channel_pos[:, 1]

        D = np.sqrt((c - a) ** 2 + (d - b) ** 2) # * 60 * 1.852
        vis_pos = channel_pos[np.argmin(D), :]
        virtual_pos[pos, :] = vis_pos

    ll['virtual_lon'] = virtual_pos[:, 0]
    ll['virtual_lat'] = virtual_pos[:, 1]
    return ll



def Get_Adjacency(batch):

    B, L, F = batch.shape

    V_f = batch[:,:, 2:4]
    A_f = np.zeros((L, B, B))
    for i in range(L):
        nodes =V_f[:, i, :]
        for f, ff in enumerate(nodes):
            a,b  = ff[0], ff[1]
            c = nodes[:, 0]
            d = nodes[:, 1]
            distance = np.sqrt(((c - a) ** 2 + (d - b) ** 2).astype('float')) * 60 # * 1.852
            A_f[i, f, :] = distance

    A_f[A_f > 1] = 0  # interaction threshold
    A_f = 1 / A_f
    A_f [A_f > 100] = 0  # transforer inf to 0

    return  A_f

def Get_Adjacency_MGSC(batch):
    B, L, F = batch.shape

    # cal A_f and A_s
    V_f = batch[:, :, 2: 4]
    V_s = np.zeros((B, L, 2))
    V_s[:,:, 0] = batch[:,:, 4] * np.sin(batch[:,:, 5].astype('float'))
    V_s[:,:, 1] = batch[:,:, 4] * np.cos(batch[:,:, 5].astype('float'))

    A_f = np.zeros((L, B, B))
    A_s = np.zeros((L, B, B))

    for s_l in range(L):
        nodes_speed = V_s[:, s_l, :]
        nodes_pos = V_f[:, s_l, :]
        Heading = np.zeros((B, 2))
        for s_i, ss in enumerate(nodes_pos):
            a, b = ss[0], ss[1]
            c = nodes_pos[:, 0]
            d = nodes_pos[:, 1]
            distance = np.sqrt(((c - a) ** 2 + (d - b) ** 2).astype('float')) * 60
            A_f[s_l, s_i, :] = distance

            distance[s_i] = 1
            Heading[: , 0] = (c - a)/ distance  # form the node s_i to others nodes
            Heading[: , 1] = (d - b)/ distance
            component_from_one = Heading[:, 0] * nodes_speed[s_i, 0] + Heading[:, 1] * nodes_speed[s_i, 1]
            component_from_otehrs = (- Heading[:, 0]) * nodes_speed[:,0] + (-Heading[:, 1]) * nodes_speed[:,1]
            A_s[s_l, s_i, :] = np.abs(component_from_one) + np.abs(component_from_otehrs)

    A_f[A_f > 1] = 0  # interaction threshold
    A_s = A_s * A_f

    A_f = 1 / A_f
    A_f[A_f > 100] = 0  # transforer inf to 0
    A_s = A_s * A_f * A_f


    # cal A_c
    V_f = batch[:, :, 2:4]
    V_c = batch[:, :, -2:]
    A_c = np.zeros((L, B, B))
    # cal distance in every actual node and all c first
    for c_l in range(L):
        nodes_actual = V_f[:, c_l, :]
        nodes_virtual = V_c[:, c_l, :]
        for c_i, cc in enumerate(nodes_actual):
            a, b = cc[0], cc[1]
            c = nodes_virtual[:, 0]
            d = nodes_virtual[:, 1]
            distance = np.sqrt(((c - a) ** 2 + (d - b) ** 2).astype('float')) * 60  # * 1.852
            A_c[c_l, c_i, :] = distance

    A_c = 1 / A_c  # for other vessel virtual pos
    for m in range(B): # for own vessel virtual pos
        A_c[:,m,m] = np.sqrt(1 / A_c[:,m,m])


    # cal A_v
    V_v = batch[:, 0, 6]
    A_v = np.zeros((B, B))
    for v, vv in enumerate(V_v):
        diff = np.abs(vv - V_v)
        A_v[v, :] = diff
    L_max = A_v.max()

    A_v[(A_v > 0) & (A_v < L_max / 3)] = 0.5
    A_v[(A_v > L_max / 3) & (A_v < L_max * 2 / 3)] = 0.75
    A_v[(A_v > L_max * 2 / 3)] = 1
    A_v = np.repeat(A_v[np.newaxis, :, :], L, 0)

    return A_f, A_s, A_c, A_v


def Batch_mass(batch_list, ship_num_ac, MGSC):

    _, L, _ = batch_list[0].shape
    Batch = np.concatenate(batch_list, axis=0)
    if MGSC:
        Batch = np.transpose(Batch[:,:, :],(1, 0, 2))
    else:
        Batch = np.transpose(Batch[:,:, 2:].astype('float32'),(1, 0, 2))
        
    # get Adjacency
    if MGSC:
        A_f = np.zeros((L, ship_num_ac, ship_num_ac))
        A_s = np.zeros((L, ship_num_ac, ship_num_ac))
        A_c = np.zeros((L, ship_num_ac, ship_num_ac))
        A_v = np.zeros((L, ship_num_ac, ship_num_ac))
        ships_in_B = 0
        for b, bb in enumerate(batch_list):
            ships_b = bb.shape[0]
            a_f, a_s, a_c, a_v = Get_Adjacency_MGSC(bb)
            A_f[:, ships_in_B: ships_in_B + ships_b, ships_in_B: ships_in_B + ships_b] = a_f
            A_s[:, ships_in_B: ships_in_B + ships_b, ships_in_B: ships_in_B + ships_b] = a_s
            A_c[:, ships_in_B: ships_in_B + ships_b, ships_in_B: ships_in_B + ships_b] = a_c
            A_v[:, ships_in_B: ships_in_B + ships_b, ships_in_B: ships_in_B + ships_b] = a_v
            ships_in_B = ships_in_B + ships_b
        pack = (Batch, A_f, A_s, A_s, A_v)
    else:
        A_f = np.zeros((L, ship_num_ac, ship_num_ac))
        ships_in_B = 0
        for b, bb in enumerate(batch_list):
            ships_b = bb.shape[0]
            a_f = Get_Adjacency(bb)
            A_f[:, ships_in_B: ships_in_B + ships_b,  ships_in_B: ships_in_B + ships_b] = a_f
            ships_in_B = ships_in_B + ships_b
        pack = (Batch, A_f)

    return pack

