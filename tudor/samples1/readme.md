# 4 gestures
1 -> yes 
2 -> no
3 -> open mouth
4 -> raise both eyebrows

# 3 types of features extracted on same video clips, grouped per folder
raw -> 478 face mesh points (x,y,z) normalized on image widht and heigh ( dependend on distance of subject to camera)
raw_normalized -> 478 face mesh points (x,y,z) normalized on distance between eyes ( no dependecy on distance between subject and camera)
processed -> 52 blendshape coeficients provided by mediapipe (mouth opened, eyebrow raised, etc)
