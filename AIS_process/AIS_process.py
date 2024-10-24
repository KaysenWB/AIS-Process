import pandas as pd
from Functions import *
import pickle
import warnings
warnings.filterwarnings('ignore')


# args
start_timestamp = '2022-09-09 08:00:00' # sampling start_timestamp of process task
end_timestamp = '2022-09-10 08:00:00'
start_time =timestamp_trans(start_timestamp)
end_time =timestamp_trans(end_timestamp)
iv = 10  # interpolation interval
report_points = 192 # report points in a sample, also time scale
sample_scale = iv * report_points # time scale for a sample/data_son (points * report interval)
sampling_interval = 30 # sampling distance in data_raw
#min_repoint = int((sample_scale / 120)) *0.8 # for a traj, the threshold we keep it
Show_Tra_Comp = False # Whether or not to show the integrity of this son's trajectory
Show_Tra = True  # Whether to visualize trajectories
MGSC =  False  # Whether or not the data is processed as a multi-graph spatial convolution
channel_root = None
Batch_number = 48 # number of ship in a mass Batch
AIS_data_root = '/Users/yangkaisen/MyProject/Data/20220908-20220930_final.csv'
save_root = '/Users/yangkaisen/MyProject/AIS_process/test.cpkl'
map_root = '/Users/yangkaisen/MyProject/Data/map/map_Aarea.png'
map_area = [114.099003, 114.187537, 22.265695, 22.322062]
channel_pos =  pd.read_csv(channel_root).values if MGSC else None



# get_ais_small() # get part of data from A month AIS data
print('Loading process')
df = pd.read_csv(AIS_data_root)
df = df[['UpdateTime (UTC)','MMSI','Longitude (deg)','Latitude (deg)', 'Speed (kn)', 'Heading (deg)', 'Length (m)']]
df = df[df['Longitude (deg)']>map_area[0]]
df = df[df['Longitude (deg)']<map_area[1]]
df = df[df['Latitude (deg)']>map_area[2]]
df = df[df['Latitude (deg)']<map_area[3]]
df = df.reset_index(drop=True)
df.sort_values( by='UpdateTime (UTC)')
df.index  = df['UpdateTime (UTC)']
df = df[start_timestamp: end_timestamp]
sample_times = int((end_time - start_time) / sampling_interval) # number of sampling

print('Start process')
ship_num_list = []
ship_num_ac = 0
batch_small = []
batch_mass = []

for i in range(sample_times):
    print(f"period{i}")

    # sample divide
    S = start_time + i * sampling_interval
    E = start_time + i * sampling_interval + sample_scale
    df_son = df [date_trans(S):date_trans(E)]

    # remove error
    df_son = Remove_Error(df_son, start_comp = S, end_comp = E)
    if i % 10 == 0 and Show_Tra:
        Show_DFson(df_son, map_root, map_area)
    if Show_Tra_Comp:
        completed, no_comp_list = Show_Tra_Completed(df_son,start_comp = S,end_comp = E)
    if df_son.empty == True:
        continue

    # insert and enrich feats
    ships_inserted =[]
    date = pd.DataFrame(pd.date_range(start=date_trans(S), periods=report_points, freq='10s'),columns=['date'])
    grouped = df_son.groupby(by='MMSI')
    for l, ll in grouped:
        ll_inserted = G_insert(ll, report_points, date)
        if MGSC:
            ll_inserted = Get_virtual_pos(ll_inserted, channel_pos)
        ships_inserted.append(ll_inserted.values)
    batch = np.stack(ships_inserted)
    if i % 10 == 0 and Show_Tra:
        Show_Batch(batch, map_root, map_area)
    ship_num = batch.shape[0]

    # mass Batch and get Adj
    if ship_num >= Batch_number:
        batch = np.transpose(batch[:, :, 2:].astype('float32'),(1, 0, 2))
        if MGSC:
            A_f, A_s, A_c, A_v = Get_Adjacency_MGSC(batch)
            Batch = (batch, A_f, A_s, A_c, A_v)
        else:
            A_f = Get_Adjacency(batch)
            Batch = (batch, A_f)
        batch_mass.append(Batch) # Batch: batch data and adj data

    else:
        ship_num_ac = ship_num_ac + ship_num
        batch_small.append(batch)
        if ship_num_ac >= Batch_number:
            Batch = Batch_mass(batch_small, ship_num_ac, MGSC)
            batch_mass.append(Batch)
            ship_num_ac = 0
            batch_small = []

# save
print('Saving')
f = open(save_root, "wb")
pickle.dump((batch_mass), f, protocol=2)
f.close()

