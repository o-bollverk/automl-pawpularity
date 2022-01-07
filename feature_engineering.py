# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 19:59:10 2021

@author: Renata Siimon
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp
from hyperopt import STATUS_OK, Trials
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import matplotlib.image as img
from random import randint
import pickle
from PIL import Image
import cv2 # pip install opencv-python
from imutils import paths  #pip install imutils

# path_to_data = "petfinder-pawpularity-score//"
path_to_data = "input//"
path_to_pics = path_to_data + "train//"

data = pd.read_csv(path_to_data + "train.csv", sep=',',  encoding='utf-8') # index_col=0,

pics = os.listdir(path_to_pics)

#-----------------------------------------

# SHOWING IMAGES:
from random import shuffle    
def load_pic(idx, path_to_pics):
    return img.imread(path_to_pics + pics[idx])

def load_pic2(idx, path_to_pics):
    return Image.open(path_to_pics + pics[idx])

def load_pic3(idx, path_to_pics, rgb):
    pic = cv2.imread(path_to_pics + pics[idx])
    if rgb==True:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB) # convert BRG to RGB
    return pic
   
def show_10_pics(path_to_pics, pics, data, col, labels):
    color_cols = ['white2', 'light_grey2', 'grey2', 'black2',
                  'brown2', 'orange2', 'green2', 'lime2', 
                  'dark_green2', 'cyan2', 'teal2', 'blue2',
                  'violet2', 'purple2', 'dark_red2', 'red2']
    if col=='random':
        pics_idx = [randint(0, len(pics)-1) for x in range(10)]
    elif col=='specific':
        pics_idx = labels
    elif col in color_cols:
        col_values = data.sort_values(by=[col], ascending = False)[col]
        pics_idx = col_values[:10].index.tolist()
    else:
        col_values = data.sort_values(by=[col])[col]
        min_values = col_values[:5].index.tolist()
        max_values = col_values[-5:].index.tolist()
        pics_idx = min_values + max_values
    for i in range(1,11):
        plt.subplot(2, 5, i)
        if col in color_cols:
            if i==1:
                plt.title(labels[0])
        elif col not in ('random', 'specific'):
            if i==1:
                plt.title(labels[0])
            elif i==6:
                plt.title(labels[1])
        pic_rgb = load_pic(pics_idx[i-1], path_to_pics)
        # plt.imshow(pic_rgb, plt.get_cmap('gray'))
        plt.imshow(pic_rgb)
        plt.axis('off')
        
# 10 random pics:
show_10_pics(path_to_pics, pics, data, 'random', None)

# 10 pics with max popularity:
most_popular = data[data['Pawpularity']==100] # ['Pawpularity'].index.tolist()
show_10_pics(path_to_pics, pics, most_popular, 'random', None)

# 10 pics with min popularity:
least_popular = data[data['Pawpularity']<3] 
show_10_pics(path_to_pics, pics, least_popular, 'random', None)

# # 10 non-random pics:
# show_10_pics(path_to_pics, pics, data, 'specific', [x for x in range(130,140)])

# One pic:
pic = load_pic(30, path_to_pics) 
plt.imshow(pic)

# Histogram of popularity values:
plt.hist(data['Pawpularity'])
plt.xlabel('Popularity score')
plt.ylabel('Number of images')
plt.title('Distribution of pet popularity')


#-----------------------------------------
# FEATURE ENGINEERING:
#-----------------------------------------

# DPI AND ASPECT RATIO (also height and width):
    
def calc_metrics(pics, path_to_pics):
    heights, widths = [], []
    aspect_ratios = []  # height/width
    DPIs = [] # density per inch (resolution) - assuming that longest dimension is 4 inches long
    # sizes = [] # measured by area (an alternative could be average of width and length.); not very informative, because images are resized on screen
    # diagonals = [] # did not give good results- it mainly distinguished images that were square vs elongated, which is also described by aspect ratio 
    for i in range(len(pics)):
        pic = load_pic(i, path_to_pics)
        
        height, width = pic.shape[0], pic.shape[1]
        aspect_ratio = height/width
        # img_size = height*width 
        # diagonal = np.sqrt(height**2 + width**2)
        dpi = int(np.max([height, width])/4)
        
        heights.append(height)
        widths.append(width)
        aspect_ratios.append(aspect_ratio) 
        # sizes.append(img_size) 
        # diagonals.append(diagonal)
        DPIs.append(dpi) 
        
    return [heights, widths, aspect_ratios, DPIs]

results = calc_metrics(pics, path_to_pics)
data['height'] = results[0] # probably better to use aspect ratio instead
data['width'] = results[1] # probably better to use aspect ratio instead
data['aspect_ratio'] = results[2]
# data['img_size'] = results[3]
# data['diagonal'] = results[3]
data['dpi'] = results[3]


# Plotting:

# Images with smallest and largest aspect ratio:
show_10_pics(path_to_pics, pics, data, 'aspect_ratio', ['Smallest aspect ratio:', 'Largest aspect ratio:']) 
# - this also shows if image is in portrait or landscale mode

# Images with smallest and largest resolution:
show_10_pics(path_to_pics, pics, data, 'dpi', ['Smallest resolution:', 'Largest resolution:']) 
# -> The usefulness of this really depends on how large images were displayed to people who evaluated the pawpularity score. 


# ---------------------

# BLUR:  
# - https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

def calc_blur(pics, path_to_pics):
    blurs = []
    for i in range(len(pics)):
        pic = load_pic3(i, path_to_pics, True)
        gray = cv2.cvtColor(pic, cv2.IMREAD_GRAYSCALE) # convert to grayscale
        blur = cv2.Laplacian(gray, cv2.CV_64F).var() # variance of laplacian
        blurs.append(blur)
        # 	if blur < 100  -> is blurry 
    return [blurs]

results = calc_blur(pics, path_to_pics)
data['blur'] = results[0]

show_10_pics(path_to_pics, pics, data, 'blur', ['Least blurry:', 'Most blurry:']) 

# --------------------------------------

# HUE, SATURATION, BRIGHTNESS   

def hsv_colors(pics, path_to_pics):
    red, yellow, green, cyan, blue, magenta = [], [], [], [], [], []
    saturation, brightnesses, hues = [], [], []
    for i in range(len(pics)):
        pic = load_pic3(i, path_to_pics, False)
        pic2 = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV) # convert  RGB to HSV
        h,s,v = cv2.split(pic2)  # (720, 405)
                
        # Hue, saturation and brightness:
        tot = s.shape[0]*s.shape[1]
        s = s.reshape(tot)
        h = h.reshape(tot)
        v = v.reshape(tot)
        # saturation.append(np.mean(s))
        saturation.append(np.mean(s)/255)
        brightnesses.append(np.mean(v)/255)
        hues.append(np.mean(h)/255)
 
    result = pd.DataFrame(data = [saturation, brightnesses, hues]).T
    result.columns = ['saturation', 'brightness', 'hue']
    return result

result = hsv_colors(pics, path_to_pics)
data = pd.concat([data, result], axis=1)

show_10_pics(path_to_pics, pics, data, 'saturation', ['smallest saturation:', 'Largest saturation:'])
show_10_pics(path_to_pics, pics, data, 'brightness', ['Darkest:', 'Brightest:']) 
show_10_pics(path_to_pics, pics, data, 'hue', ['smallest hue:', 'Largest hue:'])

# show_10_pics(path_to_pics, pics, data, 'specific', [x for x in range(10)])


# --------------------------------------

# COLORS:
    
# - mapping based on: 
# https://en.wikipedia.org/wiki/Web_colors#HTML_color_names

def map_colors2(h, s, v):  # h = hue layer
    tot = h.shape[0]*h.shape[1]
    h = h.reshape(tot)
    s = s.reshape(tot)/255
    v = v.reshape(tot)/255
     
    #hue ranges:
    h30 = h<30
    h60 = np.logical_and(h<60, h>=30)
    h90 = np.logical_and(h<90, h>=60)
    h120 = np.logical_and(h<120, h>=90)
    h150 = np.logical_and(h<150, h>=120)
    h180 = h>=150 
        
    # saturation ranges:
    s0 = s<0.5
    s100 = s>=0.5
    
    # brightness ranges:
    # v0 = v<0.25
    v50 = np.logical_and(v>=0.25, v<0.63)
    v75 = np.logical_and(v>=0.63, v<0.86)
    # v100 = v>=0.86
    v50b = v<0.5
    v100b = v>=0.5    
    
    # Combinations:
    s0_h0 = h30 & s0 # np.logical_and(h30, s0)
    s100_v50 = s100 &  v50b # np.logical_and(s100, v50b)
    s100_v100 = s100 & v100b # np.logical_and(s100, v100b)
    
    # colors:    
    white = np.sum(np.logical_and(s0_h0, v>=0.86))
    light_grey = np.sum(np.logical_and(s0_h0, v75))
    grey = np.sum(np.logical_and(s0_h0, v50))
    black = np.sum(np.logical_and(s0_h0, v<0.25))
    red = np.sum(np.logical_and(h30, s100_v50))
    maroon = np.sum(np.logical_and(h30, s100_v100))
    yellow = np.sum(np.logical_and(h60, s100_v50))
    olive = np.sum(np.logical_and(h60, s100_v100))
    lime = np.sum(np.logical_and(h90, s100_v50))
    green = np.sum(np.logical_and(h90, s100_v100))
    acqua = np.sum(np.logical_and(h120, s100_v50))
    teal = np.sum(np.logical_and(h120, s100_v100))
    blue = np.sum(np.logical_and(h150, s100_v50))
    navy = np.sum(np.logical_and(h150, s100_v100))
    fucshia = np.sum(np.logical_and(h180, s100_v50))
    purple = np.sum(np.logical_and(h180, s100_v100)) 
    
    colors_tot = np.array([white, light_grey, grey, black, red, maroon, yellow, olive, lime, green, acqua, teal, blue, navy, fucshia, purple])

    colors_perc = colors_tot/np.sum(colors_tot)

    return colors_perc


def hsv_colors2(pics, path_to_pics):
    mappings = []
    for i in range(len(pics)):
        pic = load_pic3(i, path_to_pics, False)
        pic2 = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV) # convert  RGB to HSV
        h,s,v = cv2.split(pic2)  # (720, 405)
        mapping = map_colors2(h, s, v)
        mappings.append(mapping)

    result = pd.DataFrame(data = mappings)
    result.columns = ['white', 'light_grey', 'grey', 'black',
                  'brown', 'orange', 'green', 'lime', 
                  'dark_green', 'cyan', 'teal', 'blue',
                  'violet', 'purple', 'dark_red', 'red']
    
    return result

result = hsv_colors2(pics, path_to_pics)
data = pd.concat([data, result], axis=1)


show_10_pics(path_to_pics, pics, data, 'white', ['Most white:'])
show_10_pics(path_to_pics, pics, data, 'light_grey', ['Most light grey:'])
show_10_pics(path_to_pics, pics, data, 'grey', ['Most grey:'])
show_10_pics(path_to_pics, pics, data, 'black', ['Most black:'])

show_10_pics(path_to_pics, pics, data, 'brown', ['Most brown:']) 
show_10_pics(path_to_pics, pics, data, 'orange', ['Most orange:'])
show_10_pics(path_to_pics, pics, data, 'green', ['Most green:']) 
show_10_pics(path_to_pics, pics, data, 'lime', ['Most lime:']) 

show_10_pics(path_to_pics, pics, data, 'dark_green', ['Most dark green:'])
show_10_pics(path_to_pics, pics, data, 'cyan', ['Most cyan:']) 
show_10_pics(path_to_pics, pics, data, 'teal', ['Most teal:']) 
show_10_pics(path_to_pics, pics, data, 'blue', ['Most blue:']) 

show_10_pics(path_to_pics, pics, data, 'violet', ['Most violet:'])
show_10_pics(path_to_pics, pics, data, 'purple', ['Most light purple:']) 
show_10_pics(path_to_pics, pics, data, 'dark_red', ['Most dark red:']) 
show_10_pics(path_to_pics, pics, data, 'red', ['Most red:']) 


# --------------------------------------

# WARMTH:
    
# https://forum.image.sc/t/how-to-calculate-measure-the-color-temperature-of-an-image/50359
# https://dsp.stackexchange.com/questions/8949/how-to-calculate-the-color-temperature-tint-of-the-colors-in-an-image

def calc_warmth(pics, path_to_pics):
    warmths = []
    for i in range(len(pics)):
        pic = load_pic(i, path_to_pics)
        r, g, b = pic[:,:,0], pic[:,:,1], pic[:,:,2]
        r_mean = np.mean(r)
        g_mean  =np.mean(g)
        b_mean  =np.mean(b)
        
        # Convert RGB to CIE tristimulus values
        # See also: https://en.wikipedia.org/wiki/CIE_1931_color_space
        X=(-0.14282*r_mean)+(1.54924*g_mean)+(-0.95641*b_mean)
        Y=(-0.32466*r_mean)+(1.57837*g_mean)+(-0.73191*b_mean)
        Z=(-0.68202*r_mean)+(0.77073*g_mean)+(0.56332*b_mean)
        
        # Calculate Normalized Chromacity Values
        Normx=X/(X+Y+Z)
        Normy=Y/(X+Y+Z)
        
        # compute CCT values
        n=(Normx-0.3320)/(0.1858-Normy)
        CCT=449*(n**3)+3525*(n**2)+6823.3*n+5520.33
        warmths.append(CCT)
    return warmths

result = calc_warmth(pics, path_to_pics)
data['warmth'] = result

show_10_pics(path_to_pics, pics, data, 'warmth', ['Warmest colors:', 'Coldest colors:'])


# Trim values of outliers:
# - 3 images had large negative value (all others were positive) -> trim those to zero:
a = data['warmth']<0
data['warmth'] = data['warmth'].where(-a, 0)

# - A few had very large positive values (way out of range) -> trim those to 25000
a = data['warmth']>25000
data['warmth'] = data['warmth'].where(-a, 25000)


# --------------------------------------

# CONTRAST:
    
# - can be calculated in several ways
# one way: st.dev. of grayscale image:
# https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
# another way: contrast = (Ymax-Ymin)/(Ymax+Ymin) -> using this formula, calculate contrast for small neighbourhood around each pixel, then average:
# https://stackoverflow.com/questions/57256159/how-extract-contrast-level-of-a-photo-opencv

def calc_contrast(pics, path_to_pics):
    contrasts = []
    contrasts2 = []
    for i in range(len(pics)):
        pic = load_pic3(i, path_to_pics, False)
 
        # --------------------------
        # By using standard deviation:
        pic2 = cv2.cvtColor(pic, cv2.IMREAD_GRAYSCALE) 
        # pic2 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) 
        contrast = pic2.std()
        contrasts.append(contrast)
                
        # --------------------------
        # By examining small neighbourhood around each pixel, then average:
        lab = cv2.cvtColor(pic, cv2.COLOR_BGR2LAB) 
        L, A, B=cv2.split(lab) # separate channels

        # compute minimum and maximum in 5x5 region using erode and dilate
        # - erode and dilate for NxN element are equivalent to min and max.
        kernel = np.ones((5,5), np.uint8)
        min_ = cv2.erode(L, kernel, iterations = 1)
        max_ = cv2.dilate(L, kernel, iterations = 1)
        min_ = min_.astype(np.float64) 
        max_ = max_.astype(np.float64) 
        contrast = (max_-min_)/(max_+min_) # compute local contrast
        average_contrast = np.nanmean(contrast)  # get average across whole image, ignoring nans
        contrasts2.append(average_contrast)
        
    return [contrasts, contrasts2]

result = calc_contrast(pics, path_to_pics)

data['contrast'] = result[0]
data['contrast2'] = result[1]

show_10_pics(path_to_pics, pics, data, 'contrast', ['Smallest contrast:', 'Largest contrast:'])
show_10_pics(path_to_pics, pics, data, 'contrast2', ['Smallest contrast:', 'Largest contrast:'])


#--------------------------------------

# SCATTERPLOTS OF ADDED FEATURES:

def make_scatterplot(data, cols, labels):
    for i in range(1,4):
        plt.subplot(1, 3, i)        
        plt.scatter(data[cols[i-1]], data['Pawpularity'], alpha = 0.3)
        if i==1:
            plt.ylabel('Popularity score')
        plt.xlabel(labels[i-1])
        # plt.title(labels[i-1] + " vs pawpularity")
    plt.show()

plt.rcParams['figure.figsize'] = (14.0, 7.0)

# Aspect ratio, DPI and blur:
make_scatterplot(data, cols=['aspect_ratio', 'dpi', 'blur'], 
                 labels = ['Aspect ratio', 'DPI', 'Blur'])

# Hue, saturation and brightness:
make_scatterplot(data, cols=['hue', 'saturation', 'brightness'], 
                 labels = ['Hue', 'Saturation', 'Brightness'])

# Warmth and contrast:
make_scatterplot(data, cols=['warmth', 'contrast', 'contrast2'], 
                 labels = ['Warmth', 'Contrast', 'Contrast2'])


#  # Aspect ratio:
# plt.scatter(data['aspect_ratio'], data['Pawpularity'], alpha = 0.4)
# plt.ylabel('Popularity score')
# plt.xlabel('Aspect ratio')
# plt.title('Aspect ratio vs pawpularity')

# # DPI:
# plt.scatter(data['dpi'], data['Pawpularity'], alpha = 0.4)
# plt.ylabel('Popularity score')
# plt.xlabel('DPI')
# plt.title('DPI vs pawpularity')

# # Blur:
# plt.scatter(data['blur'], data['Pawpularity'], alpha = 0.3)
# plt.ylabel('Popularity score')
# plt.xlabel('Blur')
# plt.title('Blur vs pawpularity')

# # Hue:
# plt.scatter(data['hue'], data['Pawpularity'], alpha = 0.3)
# plt.ylabel('Popularity score')
# plt.xlabel('Hue')
# plt.title('Hue vs pawpularity')
    
# # Saturation:
# plt.scatter(data['saturation'], data['Pawpularity'], alpha = 0.3)
# plt.ylabel('Popularity score')
# plt.xlabel('Saturation')
# plt.title('Saturation vs pawpularity')
    
# # Brightness:
# plt.scatter(data['brightness'], data['Pawpularity'], alpha = 0.3)
# plt.ylabel('Popularity score')
# plt.xlabel('Brightness')
# plt.title('Brightness vs pawpularity')
    
# # Warmth:
# plt.scatter(data['warmth'], data['Pawpularity'], alpha = 0.2)
# plt.ylabel('Popularity score')
# plt.xlabel('Warmth')
# plt.title('Warmth vs pawpularity')

# # Contrast:
# plt.scatter(data['contrast'], data['Pawpularity'], alpha = 0.3)
# plt.ylabel('Popularity score')
# plt.xlabel('Contrast')
# plt.title('Contrast vs pawpularity')
    
# # Contrast2:
# plt.scatter(data['contrast2'], data['Pawpularity'], alpha = 0.3)
# plt.ylabel('Popularity score')
# plt.xlabel('Contrast2')
# plt.title('Contrast2 vs pawpularity')
    
#--------------------------------------

# Remove height, width:
data = data.drop(columns=['height', 'width'])

# Reorder columns:
data = data[['Id', 'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur', 'hue', 'saturation', 'brightness', 'contrast', 'contrast2', 'blur', 'warmth', 'aspect_ratio', 'dpi', 'white', 'light_grey', 'grey', 'black', 'brown', 'orange', 'green', 'lime', 'dark_green', 'cyan', 'teal', 'blue', 'violet', 'purple', 'dark_red', 'red', 'Pawpularity']]


#-----------------------------------------

# SAVE DATA WITH ADDED FEATURES:
data.to_csv(path_to_data + "train2.csv", sep=',', encoding = "UTF-8", index = False)
    
#-----------------------------------------

###READING RESULTS FROM FILE: 
# path_to_data = "petfinder-pawpularity-score//"
data = pd.read_csv(path_to_data + "train2.csv", delimiter=',', encoding = "UTF-8")
    
# -------------------------------------------------------

# SOME TRIALS:
    
# BASELINE:
def make_regression(X,y):
    regr = RandomForestRegressor(random_state=0)
    # regr.fit(trainX, train_y)
    # preds = regr.predict(testX)
    # rmse = mean_squared_error(test_y, preds, squared = False)  # 20.53
    
    def mean_error(test_y, preds):
        # assuming y and y_pred are numpy arrays
        rmse = mean_squared_error(test_y, preds, squared = False)
        return -rmse
    
    mean_error_scorer = make_scorer(mean_error, greater_is_better=False)
    scores = cross_val_score(regr, X, y, scoring=mean_error_scorer)
    # [-20.88841628, -20.98779094, -21.02254736, -20.34309572,
    #        -20.60567174]
    
    return scores.mean() #  -20.769 (1- perfect fit, 0- worst)
    

y = data['Pawpularity']

# Existing features:
X = data[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']] 
print(make_regression(X,y)) # 20.76950440932938

# Added features (except colors):
X = data[['hue', 'saturation', 'brightness', 'contrast', 'contrast2', 'blur', 'warmth', 'aspect_ratio', 'dpi']]  
print(make_regression(X,y)) # 21.020323605693214

# Added features (only colors):
X = data[['white', 'light_grey', 'grey', 'black', 'brown', 'orange', 'green', 'lime', 'dark_green', 'cyan', 'teal', 'blue', 'violet', 'purple', 'dark_red', 'red']]
print(make_regression(X,y)) # 20.907879530446497
    
# All added features (colors and others):
X = data[['hue', 'saturation', 'brightness', 'contrast', 'contrast2', 'blur', 'warmth', 'aspect_ratio', 'dpi' , 'white', 'light_grey', 'grey', 'black', 'brown', 'orange', 'green', 'lime', 'dark_green', 'cyan', 'teal', 'blue', 'violet', 'purple', 'dark_red', 'red']]
print(make_regression(X,y)) # 20.873154066943037
    
# Both existing and added features:
X = data.drop(columns = ['Pawpularity', 'Id'])
print(make_regression(X,y)) # 20.854336209859447

# ---------------------------------------------

# # Normaliseerimine - RF puhul ei anna efekti
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

# ---------------------------------------------

# FEATURE IMPORTANCES (without cross-validation):
    
def get_feature_importances(X, data):
    trainX, testX = X[:8000], X[8000:]
    train_y, test_y = data['Pawpularity'][:8000], data['Pawpularity'][8000:]
    regr = RandomForestRegressor(random_state=0)
    regr.fit(trainX, train_y)
    preds = regr.predict(testX)
    rmse = mean_squared_error(test_y, preds, squared = False)  # 20.987398005762916
    feat_importances = regr.feature_importances_
    feats = trainX.columns.tolist()
    feats_df = pd.DataFrame(data = [feats, feat_importances]).T
    feats_df.columns = ['Feature', 'Importance']
    feats_df = feats_df.sort_values(by = ['Importance'], ascending = False)
    return feats_df

# Existing features:
X = data[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']] 
print(get_feature_importances(X, data))
#           Feature Importance
# 3            Near   0.132239
# 6           Group   0.124146
# 9       Occlusion   0.110609
# 8           Human   0.106222
# 2            Face   0.096617
# 5       Accessory   0.088605
# 7         Collage   0.088175
# 10           Info   0.085501
# 1            Eyes   0.052946
# 4          Action    0.05256
# 11           Blur   0.038223
# 0   Subject Focus   0.024157

# Added features (except colors):
X = data[['hue', 'saturation', 'brightness', 'contrast', 'contrast2', 'blur', 'warmth', 'aspect_ratio', 'dpi']]  
print(get_feature_importances(X, data))
#         Feature Importance
# 0           hue   0.132144
# 2    brightness   0.131151
# 5          blur   0.130146
# 4     contrast2   0.128458
# 3      contrast   0.123951
# 1    saturation   0.122927
# 6        warmth   0.119621
# 7  aspect_ratio   0.071531
# 8           dpi   0.040071

# Added features (only colors):
X = data[['white', 'light_grey', 'grey', 'black', 'brown', 'orange', 'green', 'lime', 'dark_green', 'cyan', 'teal', 'blue', 'violet', 'purple', 'dark_red', 'red']]
print(get_feature_importances(X, data))
#        Feature Importance
# 1   light_grey   0.089426
# 0        white   0.088249
# 4        brown   0.086946
# 3        black   0.086161
# 6        green    0.08397
# 2         grey   0.082737
# 5       orange   0.076948
# 14    dark_red   0.076324
# 10        teal   0.072843
# 12      violet   0.067403
# 8   dark_green   0.066356
# 15         red   0.040581
# 11        blue   0.031565
# 7         lime   0.027362
# 9         cyan   0.013532
# 13      purple   0.009598

# All added features (colors and others):
X = data[['hue', 'saturation', 'brightness', 'contrast', 'contrast2', 'blur', 'warmth', 'aspect_ratio', 'dpi' , 'white', 'light_grey', 'grey', 'black', 'brown', 'orange', 'green', 'lime', 'dark_green', 'cyan', 'teal', 'blue', 'violet', 'purple', 'dark_red', 'red']]
print(get_feature_importances(X, data))
#          Feature Importance
# 5           blur   0.056016
# 10    light_grey   0.054713
# 4      contrast2   0.054205
# 9          white   0.052894
# 13         brown   0.051618
# 2     brightness   0.051324
# 15         green   0.051009
# 0            hue   0.050499
# 14        orange   0.048486
# 11          grey   0.048254
# 12         black   0.047954
# 3       contrast   0.047561
# 23      dark_red   0.045246
# 6         warmth   0.043219
# 1     saturation    0.04316
# 19          teal   0.042153
# 21        violet   0.042114
# 17    dark_green    0.03842
# 7   aspect_ratio   0.034226
# 24           red   0.023673
# 8            dpi   0.021228
# 20          blue   0.019927
# 16          lime   0.015937
# 18          cyan   0.009264
# 22        purple   0.006897

# Both existing and added features:
X = data.drop(columns = ['Pawpularity', 'Id'])
print(get_feature_importances(X, data))
#           Feature Importance
# 17           blur   0.055055
# 16      contrast2   0.052411
# 22     light_grey   0.052212
# 25          brown    0.05064
# 21          white    0.05047
# 14     brightness   0.048909
# 27          green   0.048552
# 12            hue   0.047683
# 23           grey    0.04732
# 26         orange   0.046307
# 15       contrast   0.046101
# 24          black   0.045903
# 35       dark_red   0.044024
# 13     saturation   0.041786
# 18         warmth   0.041337
# 31           teal    0.04058
# 33         violet   0.039792
# 29     dark_green   0.036703
# 19   aspect_ratio   0.032346
# 36            red   0.022763
# 20            dpi   0.020287
# 32           blue   0.019217
# 28           lime   0.015494
# 30           cyan   0.009081
# 34         purple   0.006695
# 1            Eyes   0.005073
# 9       Occlusion   0.003981
# 6           Group   0.003881
# 8           Human   0.003843
# 3            Near    0.00355
# 5       Accessory   0.003238
# 2            Face   0.003164
# 11           Blur   0.002686
# 7         Collage   0.002524
# 10           Info   0.002378
# 0   Subject Focus   0.002336
# 4          Action   0.001676


#-----------------------------------------

# HYPEROPT:
def hyperopt_train_test(params):
    model = RandomForestRegressor(**params)
    mean_error_scorer = make_scorer(mean_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, scoring=mean_error_scorer)
    return scores.mean()

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,7))
}

def f(params):
    rmse = hyperopt_train_test(params)
    # return {'loss': -acc, 'status': STATUS_OK}
    return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, 
            space4rf, 
            algo=tpe.suggest, 
            max_evals=20, trials=trials)

print('best:', best)  # best: {'max_depth': 3}
# X=data[['blur', 'aspect_ratio', 'dpi', 'brightness']]  best: 20.536772

    
