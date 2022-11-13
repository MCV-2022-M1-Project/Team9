import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import os
import shutil
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances

# We cluster by the mean brightness of each image
def cluster_by_brightness(images, root_folder):
    brightness_folder = root_folder + '/brightness/'
    brightness_bright_folder = brightness_folder + 'bright/'
    brightness_dark_folder = brightness_folder + 'dark/'
    
    
    if os.path.isdir(brightness_folder):
        shutil.rmtree(brightness_folder)
    
    os.makedirs(brightness_folder)
    os.makedirs(brightness_bright_folder)
    os.makedirs(brightness_dark_folder)
    
    # Save index of image on cluster
    cluster = {
        'bright': [],
        'dark': []
    }
    
    mean_grays = []
    
    for value in images:
        im = value['image']
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mean_grays.append(np.mean(im_gray))
    
    mean_grays = np.array(mean_grays).reshape(-1,1)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(mean_grays)
    cluster_centers = kmeans.cluster_centers_
    cluster_mean = np.mean(kmeans.cluster_centers_)
    print(cluster_mean)
    # print(kmeans.get_params())
    
    # sys.exit()

    
    for index, value in enumerate(images):
        filename = value['filename']
        im = value['image']

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(im_gray)
        
        if kmeans.labels_[index] == 0:
            cluster['bright'].append(mean_gray)
            cv2.imwrite(brightness_bright_folder + os.path.basename(filename).split('.')[0] + '.png', im_gray)
            cv2.imwrite(brightness_bright_folder + os.path.basename(filename).split('.')[0] + '.jpg', im)
        else:
            cluster['dark'].append(mean_gray)
            cv2.imwrite(brightness_dark_folder + os.path.basename(filename).split('.')[0] + '.png', im_gray)
            cv2.imwrite(brightness_dark_folder + os.path.basename(filename).split('.')[0] + '.jpg', im)
        
        
    mean_grays = np.array(mean_grays)
    
    plt.hist([cluster['bright'], cluster['dark']], density=True, stacked=True, color=['Black', 'Lightgray'], label=['Dark', 'Bright'])
    plt.legend(loc='best')
    plt.axvline(cluster_mean, color='firebrick', linestyle='dashed', linewidth=1)
    plt.title('Brightness - KMeans cluster')
    plt.ylabel('Probability')
    plt.xlabel('Brightness')
    plt.show()

# https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
def im_colourfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

# Cluster by room based on some characteristic (pleasure, arousal or dominance)
def cluster_by_room(images, root_folder, characteristic = 'pleasure'):
    
    rooms_folder = root_folder + characteristic + '/rooms/'
    
    if os.path.isdir(rooms_folder):
        shutil.rmtree(rooms_folder)
    
    for i in range(5):
        os.makedirs(rooms_folder + str(i))
    
    
    all_images_characteristic = []
    all_images = []
    
    X = []
    
    for index, value in enumerate(images):
        im = value['image']
        
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        mean_saturation = np.mean(im_hsv[:,:,1])
        mean_brightness = np.mean(im_hsv[:,:,2])
        
        if characteristic == 'pleasure':
            characteristic_value = mean_brightness * 0.69 + mean_saturation * 0.22
        elif characteristic == 'arousal':
            characteristic_value = mean_brightness * (-0.31) + mean_saturation * 0.60
        elif characteristic == 'dominance':
            characteristic_value = mean_brightness * 0.76 + mean_saturation * 0.32
        
        images[index]['characteristic'] = characteristic_value
        
        all_images_characteristic.append(characteristic_value)
        all_images.append(im)
        
        X.append([characteristic_value])
    
    X = np.array(X)
    
    kmeans = KMeans(n_clusters=5, random_state=0)
    y_pred = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    
    for index, label in enumerate(y_pred):
        filename = rooms_folder + str(label) + '/' + os.path.basename(images[index]['filename'])
        cv2.imwrite(filename, images[index]['image'])
    
    colors = ListedColormap(['r','b','y','g','c'])
    classes = ['0', '1', '2', '3', '4']
    
    fig = plt.figure(figsize=(12, 12))
    
    clusters = []
    
    for i in range(5):
        clusters.append(np.take(all_images_characteristic, np.where(y_pred == i)).flatten())
    
    
    plt.hist(clusters, density=True, stacked=True, color=['r','b','y','g','c'], label=classes)
    plt.legend(loc='best')
    for centroid in centroids:
        plt.axvline(centroid, color='firebrick', linestyle='dashed', linewidth=1)
    plt.title(characteristic.capitalize() + ' - KMeans cluster')
    plt.ylabel('Probability')
    plt.xlabel(characteristic)
    plt.show()

# Cluster by room based on the 3 pad (pleasure, arousal or dominance) axis that create a new space
def cluster_by_room_pad(images, root_folder):
    
    rooms_folder = root_folder + 'rooms/'
    
    if os.path.isdir(rooms_folder):
        shutil.rmtree(rooms_folder)
    
    for i in range(5):
        os.makedirs(rooms_folder + str(i))
    
    
    all_images_pleasure = []
    all_images_arousal = []
    all_images_dominance = []
    all_images = []
    
    X = []
    
    for index, value in enumerate(images):
        im = value['image']
        
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_canny = cv2.Canny(im_gray, 50, 150, None, 3)
        
        mean_saturation = np.mean(im_hsv[:,:,1])
        mean_brightness = np.mean(im_hsv[:,:,2])
        
        image_pleasure = mean_brightness * 0.69 + mean_saturation * 0.22
        image_arousal = mean_brightness * (-0.31) + mean_saturation * 0.60
        image_dominance = mean_brightness * 0.76 + mean_saturation * 0.32
        
        images[index]['pleasure'] = image_pleasure
        images[index]['arousal'] = image_arousal
        images[index]['dominance'] = image_dominance
        
        all_images_pleasure.append(image_pleasure)
        all_images_arousal.append(image_arousal)
        all_images_dominance.append(image_dominance)
        all_images.append(im)
        
        X.append([image_pleasure, image_arousal, image_dominance])
    
    X = np.array(X)
    
    kmeans = KMeans(n_clusters=5, random_state=0)
    y_pred = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    for index, label in enumerate(y_pred):
        filename = rooms_folder + str(label) + '/' + os.path.basename(images[index]['filename'])
        cv2.imwrite(filename, images[index]['image'])
        
    distances = pairwise_distances(centroids, X, metric='euclidean')
    ind = [np.argpartition(i, 5)[:5] for i in distances]
    # closest = [X[indexes] for indexes in ind]
    
    for label, cluster_indexes in enumerate(ind):
        os.mkdir(rooms_folder + str(label) + '/representative')
        for index in cluster_indexes:
            filename = rooms_folder + str(label) + '/representative/' + os.path.basename(images[index]['filename'])
            cv2.imwrite(filename, images[index]['image'])
    
    
    colors = ListedColormap(['r','b','y','g','c'])
    classes = ['0', '1', '2', '3', '4']
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(all_images_pleasure, all_images_arousal, all_images_dominance, c=y_pred, cmap=colors)
    # scatter = plt.scatter(all_images_pleasure, all_images_arousal, all_images_dominance, c=y_pred, cmap=colors)
    ax.legend(handles=scatter.legend_elements()[0], labels=classes)

    for index, centroid in enumerate(centroids):
        print(centroid)
        ax.scatter(centroid[0],centroid[1],centroid[2], color='m', marker='x', s=150, linewidths = 5, zorder = 10)
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    plt.show()
    
    
root_folder = './Datasets/BBDD/'
images = []

for file in glob.glob(root_folder + '*.jpg'):
    filename = root_folder + os.path.basename(file).split('.')[0] + '.jpg'
    
    im = cv2.imread(file)  
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    images.append({'filename': filename, 'image':im})

# Clustering
cluster_by_brightness(images, root_folder)
cluster_by_room(images, root_folder, 'pleasure')
cluster_by_room(images, root_folder, 'arousal')
cluster_by_room(images, root_folder, 'dominance')
cluster_by_room_pad(images, root_folder)