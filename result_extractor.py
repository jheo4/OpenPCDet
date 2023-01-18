import re

data = \
"""
2023-01-09 03:14:37,831   INFO  recall_roi_0.3: 0.000000
2023-01-09 03:14:37,831   INFO  recall_rcnn_0.3: 0.667559
2023-01-09 03:14:37,831   INFO  recall_roi_0.5: 0.000000
2023-01-09 03:14:37,831   INFO  recall_rcnn_0.5: 0.508429
2023-01-09 03:14:37,831   INFO  recall_roi_0.7: 0.000000
2023-01-09 03:14:37,831   INFO  recall_rcnn_0.7: 0.235790
2023-01-09 03:14:37,832   INFO  Average predicted number of objects(3769 samples): 9.856
2023-01-09 03:14:48,901   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:68.6880, 51.8560, 48.4364
bev  AP:49.3588, 37.8046, 35.6815
3d   AP:33.3140, 27.2230, 24.2427
aos  AP:67.97, 51.06, 47.54
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:68.6463, 50.4862, 48.0234
bev  AP:47.3380, 35.6754, 32.3223
3d   AP:30.5043, 23.1579, 20.4720
aos  AP:67.91, 49.62, 46.97
Car AP@0.70, 0.50, 0.50:
bbox AP:68.6880, 51.8560, 48.4364
bev  AP:84.7508, 66.3056, 63.1953
3d   AP:80.8349, 62.1908, 58.3688
aos  AP:67.97, 51.06, 47.54
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:68.6463, 50.4862, 48.0234
bev  AP:86.1838, 66.4663, 62.8162
3d   AP:82.6627, 61.8985, 57.8709
aos  AP:67.91, 49.62, 46.97
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:17.9631, 15.3734, 15.0708
bev  AP:7.6974, 5.7631, 5.1534
3d   AP:5.2874, 4.3184, 4.0659
aos  AP:9.75, 8.96, 8.59
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:16.0405, 13.9936, 13.0203
bev  AP:5.5547, 4.3382, 3.6400
3d   AP:3.4483, 2.4971, 2.0518
aos  AP:8.54, 7.64, 6.95
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:17.9631, 15.3734, 15.0708
bev  AP:21.1717, 20.2951, 18.1152
3d   AP:20.5429, 18.7502, 17.4694
aos  AP:9.75, 8.96, 8.59
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:16.0405, 13.9936, 13.0203
bev  AP:19.7983, 18.0180, 16.6025
3d   AP:18.9436, 16.9929, 15.5251
aos  AP:8.54, 7.64, 6.95
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:9.5730, 9.3876, 9.4331
bev  AP:9.0909, 9.0909, 9.0909
3d   AP:4.5455, 4.5455, 4.5455
aos  AP:9.44, 9.29, 9.33
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:2.1991, 0.9153, 1.1740
bev  AP:0.8136, 0.3987, 0.5177
3d   AP:0.3530, 0.1565, 0.2689
aos  AP:1.88, 0.63, 0.82
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:9.5730, 9.3876, 9.4331
bev  AP:9.2885, 9.0909, 9.0909
3d   AP:9.2812, 9.0909, 9.0909
aos  AP:9.44, 9.29, 9.33
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:2.1991, 0.9153, 1.1740
bev  AP:1.0888, 0.5685, 0.7426
3d   AP:1.0321, 0.5625, 0.7328
aos  AP:1.88, 0.63, 0.82
"""

pattern = re.compile(r'recall_rcnn_\d\.\d: (\d\.\d+)')
matches = pattern.finditer(data)

for match in matches:
    print(match.group(0))
# split the input into lines
lines = data.split("\n")

# loop through the lines and find the lines that match the desired output
for line in lines:
    match = re.search("(Car|Pedestrian|Cyclist) AP*", line)
    if match:
        car_res = re.search("(.* Car )(.*)", line)
        if car_res:
            print(car_res.group(2))
        else:
            print(line)
        continue

    match = re.search("3d   AP:[0-9.]+", line)
    if match:
        print(line)

