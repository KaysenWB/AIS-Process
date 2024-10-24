# AIS-Process
This is an AIS processing file containing the following tasks: trajectory cleaning, interpolation, making adjacency matrices, packing multiple multi-vessel scenarios and trajectory visualisation. The required input is a csv file, the columns of the file include but are not limited to: [‘UpdateTime (UTC)’, ‘MMSI’, ‘Longitude (deg)’, ‘Latitude (deg)’, ‘Speed (kn)’, ‘Heading (deg)’, ‘Length (m)’]. If MGSC=False, the output is the a Batch of trajectory data [length, nodes, features], and a adjacency matrix [length, nodes, nodes] packed with multiple scenarios. If MGSC=True, the output is a Batch of trajectory data, and four adjacency matrices corresponding to the vessel position, the speed and heading, the virtual position of the channel, and the size of the vessel (for details refer to: Graph-driven multi-vessel long-term trajectories prediction for route planning under complex waters).

![AIS_clean](https://github.com/KaysenWB/AIS-Process/blob/main/AIS_clean.jpg?raw=true)
![AIS_interpolated](https://github.com/KaysenWB/AIS-Process/blob/main/AIS_interpolated.jpg?raw=true)


