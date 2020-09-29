#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFont, ImageColor
import matplotlib.pyplot as plt
import random
from pygraphviz import *
import cv2
from torchvision.utils import save_image
import networkx as nx
import copy
#from intersections import doIntersect
import svgwrite

EXP_ID = random.randint(0, 1000000)

#     labelMap['living_room'] = 1
#     labelMap['kitchen'] = 2
#     labelMap['bedroom'] = 3
#     labelMap['bathroom'] = 4
#     labelMap['restroom'] = 4
#     labelMap['washing_room'] = 4    
#     labelMap['office'] = 3
#     labelMap['closet'] = 6
#     labelMap['balcony'] = 7
#     labelMap['corridor'] = 8
#     labelMap['dining_room'] = 9
#     labelMap['laundry_room'] = 10
#     labelMap['PS'] = 10   

# ORIGINAL HOUSE-GAN
# ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8,
#               "dining_room": 9, "laundry_room": 10}
# CLASS_ROM = {}
# for x, y in ROOM_CLASS.items():
#     CLASS_ROM[y] = x
# ID_COLOR = {1: 'brown', 2: 'magenta', 3: 'orange', 4: 'gray', 5: 'red', 6: 'blue', 7: 'cyan', 8: 'green', 9: 'salmon', 10: 'yellow'}




ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "corridor": 6, "dinning room": 7, "study": 8,
              "studio": 9, "store room": 10,  "graden":11 , "laundry room" :12, "office":13, "basement":14, "garage":15, "undefined":16, "door":17, "norinth":18 }

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
import webcolors
ID_COLOR = {1: 'orangered', 2: 'mediumseagreen', 3: 'yellow', 4: 'blue', 5: 'orange', 6: 'purple', 7: 'skyblue', 8: 'magenta', 9:'greenyellow', 10:'salmon', 11: 'darkslategrey',18:'white', 12: 'plum',13: 'sienna' ,14: 'papayawhip' ,15: 'maroon' ,16: 'lightgreen', 17:'olive' }


NUM_WALL_CORNERS = 13
NUM_CORNERS = 21
#CORNER_RANGES = {'wall': (0, 13), 'opening': (13, 17), 'icon': (17, 21)}

NUM_ICONS = 7
NUM_ROOMS = 10
POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )], [(0, 3), (0, 1), (1, 2), (2, 3)], [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], [(0, 1, 2, 3)]]

class ColorPalette:
    def __init__(self, numColors):
        #np.random.seed(2)
        #self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0

        
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],                                   
                                  [255, 255, 0],                                  
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],                                  
                                  [0, 100, 100],
                                  [0, 255, 128],                                  
                                  [0, 128, 255],
                                  [255, 0, 128],                                  
                                  [128, 0, 255],
                                  [255, 128, 0],                                  
                                  [128, 255, 0],                                                                    
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.random.randint(255, size = (numColors, 3))
            pass
        
        return

    def getColorMap(self):
        return self.colorMap
    
    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass
        return

def isManhattan(line, gap=3):
    return min(abs(line[0][0] - line[1][0]), abs(line[0][1] - line[1][1])) < gap

def calcLineDim(points, line):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    if abs(point_2[0] - point_1[0]) > abs(point_2[1] - point_1[1]):
        lineDim = 0
    else:
        lineDim = 1
        pass
    return lineDim

def calcLineDirection(line, gap=3):
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))


## Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))


def drawWallMask(walls, width, height, thickness=3, indexed=False):
    if indexed:
        wallMask = np.full((height, width), -1, dtype=np.int32)
        for wallIndex, wall in enumerate(walls):
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=wallIndex, thickness=thickness)
            continue
    else:
        wallMask = np.zeros((height, width), dtype=np.int32)
        for wall in walls:
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=1, thickness=thickness)
            continue
        wallMask = wallMask.astype(np.bool)
        pass
    return wallMask


def extractCornersFromHeatmaps(heatmaps, heatmapThreshold=0.5, numPixelsThreshold=5, returnRanges=True):
    """Extract corners from heatmaps"""
    from skimage import measure 
    print(heatmaps.shape)
    
    heatmaps = (heatmaps > heatmapThreshold).astype(np.float32)
    orientationPoints = []
    #kernel = np.ones((3, 3), np.float32)
    for heatmapIndex in range(0, heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, heatmapIndex]
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min() + 1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            if ys.shape[0] <= numPixelsThreshold:
                continue
            #print(heatmapIndex, xs.shape, ys.shape, componentIndex)
            if returnRanges:
                points.append(((xs.mean(), ys.mean()), (xs.min(), ys.min()), (xs.max(), ys.max())))
            else:
                points.append((xs.mean(), ys.mean()))
                pass
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def extractCornersFromSegmentation(segmentation, cornerTypeRange=[0, 13]):
    """Extract corners from segmentation"""
    from skimage import measure
    orientationPoints = []
    for heatmapIndex in range(cornerTypeRange[0], cornerTypeRange[1]):
        heatmap = segmentation == heatmapIndex
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min()+1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            points.append((xs.mean(), ys.mean()))
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def getOrientationRanges(width, height):
    orientationRanges = [[width, 0, 0, 0], [width, height, width, 0], [width, height, 0, height], [0, height, 0, 0]]
    return orientationRanges

def getIconNames():
    iconNames = []
    iconLabelMap = getIconLabelMap()
    for iconName, _ in iconLabelMap.items():
        iconNames.append(iconName)
        continue
    return iconNames

def getIconLabelMap():
    labelMap = {}
    labelMap['bathtub'] = 1
    labelMap['cooking_counter'] = 2
    labelMap['toilet'] = 3
    labelMap['entrance'] = 4
    labelMap['washing_basin'] = 5
    labelMap['special'] = 6
    labelMap['stairs'] = 7
    labelMap['door'] = 8
    return labelMap


def drawPoints(filename, width, height, points, backgroundImage=None, pointSize=5, pointColor=None):
  colorMap = ColorPalette(NUM_CORNERS).getColorMap()
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 3), np.uint8)
  else:
    if backgroundImage.ndim == 2:
      image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 3])
    else:
      image = backgroundImage
      pass
  pass
  no_point_color = pointColor is None
  for point in points:
    if no_point_color:
        pointColor = colorMap[point[2] * 4 + point[3]]
        pass
    #print('used', pointColor)
    #print('color', point[2] , point[3])
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width)] = pointColor
    continue

  if filename != '':
    cv2.imwrite(filename, image)
    return
  else:
    return image

def drawPointsSeparately(path, width, height, points, backgroundImage=None, pointSize=5):
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 13), np.uint8)
  else:
    image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 13])
    pass

  for point in points:
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width), int(point[2] * 4 + point[3])] = 255
    continue
  for channel in range(13):
    cv2.imwrite(path + '_' + str(channel) + '.png', image[:, :, channel])
    continue
  return

def drawLineMask(width, height, points, lines, lineWidth = 5, backgroundImage = None):
  lineMask = np.zeros((height, width))

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)

    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(min(point_1[direction], point_2[direction]))
    maxValue = int(max(point_1[direction], point_2[direction]))
    if direction == 0:
      lineMask[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1] = 1
    else:
      lineMask[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width)] = 1
      pass
    continue
  return lineMask



def drawLines(filename, width, height, points, lines, lineLabels = [], backgroundImage = None, lineWidth = 5, lineColor = None):
  colorMap = ColorPalette(len(lines)).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    if backgroundImage.ndim == 2:
      image = np.stack([backgroundImage, backgroundImage, backgroundImage], axis=2)
    else:
      image = backgroundImage
      pass
    pass

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)


    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(round(min(point_1[direction], point_2[direction])))
    maxValue = int(round(max(point_1[direction], point_2[direction])))
    if len(lineLabels) == 0:
      if np.any(lineColor == None):
        lineColor = np.random.rand(3) * 255
        pass
      if direction == 0:
        image[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1, :] = lineColor
      else:
        image[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width), :] = lineColor
    else:
      labels = lineLabels[lineIndex]
      isExterior = False
      if direction == 0:
        for c in range(3):
          image[max(fixedValue - lineWidth, 0):min(fixedValue, height), minValue:maxValue, c] = colorMap[labels[0]][c]
          image[max(fixedValue, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue, c] = colorMap[labels[1]][c]
          continue
      else:
        for c in range(3):
          image[minValue:maxValue, max(fixedValue - lineWidth, 0):min(fixedValue, width), c] = colorMap[labels[1]][c]
          image[minValue:maxValue, max(fixedValue, 0):min(fixedValue + lineWidth + 1, width), c] = colorMap[labels[0]][c]
          continue
        pass
      pass
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)


def drawRectangles(filename, width, height, points, rectangles, labels, lineWidth = 2, backgroundImage = None, rectangleColor = None):
  colorMap = ColorPalette(NUM_ICONS).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    image = backgroundImage
    pass

  for rectangleIndex, rectangle in enumerate(rectangles):
    point_1 = points[rectangle[0]]
    point_2 = points[rectangle[1]]
    point_3 = points[rectangle[2]]
    point_4 = points[rectangle[3]]


    if len(labels) == 0:
      if rectangleColor is None:
        color = np.random.rand(3) * 255
      else:
        color = rectangleColor
    else:
      color = colorMap[labels[rectangleIndex]]
      pass

    x_1 = int(round((point_1[0] + point_3[0]) / 2))
    x_2 = int(round((point_2[0] + point_4[0]) / 2))
    y_1 = int(round((point_1[1] + point_2[1]) / 2))
    y_2 = int(round((point_3[1] + point_4[1]) / 2))

    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=tuple(color.tolist()), thickness = 2)
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)
    pass

def pointDistance(point_1, point_2):
    #return np.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))

def calcLineDirectionPoints(points, line):
  point_1 = points[line[0]]
  point_2 = points[line[1]]
  if isinstance(point_1[0], tuple):
      point_1 = point_1[0]
      pass
  if isinstance(point_2[0], tuple):
      point_2 = point_2[0]
      pass
  return calcLineDirection((point_1, point_2))

def open_png(im_path, im_size=512):
	
	# open graph image
	png = Image.open(im_path)
	im = Image.new("RGB", png.size, (255, 255, 255))
	im.paste(png, mask=png.split()[3])
	w, h = im.size
	
    # pad graph images
	a = h/w
	if w > h:
		n_w = im_size
		n_h = int(a*n_w)
	else:
		n_h = im_size
		n_w = int(n_h/a)
	im = im.resize((n_w, n_h))
	delta_w = im_size - n_w
	delta_h = im_size - n_h
	padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	im = ImageOps.expand(im, padding, fill='white').convert('RGBA')
	im_arr = np.array(im)
	
	return im_arr

# def draw_graph(nds, eds, shift, im_size=128):

#     # Create graph
#     graph = AGraph(strict=False, directed=False)

#     # Create nodes
#     for k in range(nds.shape[0]):
#         nd = np.where(nds[k]==1)[0]
#         if len(nd) > 0:
#             color = ID_COLOR[nd[0]+1]
#             name = '' #CLASS_ROM[nd+1]
#             graph.add_node(k, label=name, color=color)

#     # Create edges
#     for i, p, j in eds:
#         if p > 0:
#             graph.add_edge(i-shift, j-shift, color='black', penwidth='4')
    
#     graph.node_attr['style']='filled'
#     graph.layout(prog='neato')
#     graph.draw('temp/_temp_{}.png'.format(EXP_ID))

#     # Get array
#     png_arr = open_png('temp/_temp_{}.png'.format(EXP_ID), im_size=im_size)
#     im_graph_tensor = torch.FloatTensor(png_arr.transpose(2, 0, 1)/255.0)
#     return im_graph_tensor

def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGBA', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_graph(nds, tps, shift):
    # build true graph 
    nds = np.where(nds.copy()==1)[-1]
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(nds):
        _type = label+1 
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':_type})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in tps:
        if m > 0:
            G_true.add_edges_from([(k-shift, l-shift)], color='b',weight=4)    
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u,v in edges]
    weights = [4 for u, v in edges]

    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold', edges=edges, edge_color=colors, width=weights)
    plt.tight_layout()
    plt.savefig('./temp/{}.jpg'.format(EXP_ID), format="jpg")
    rgb_im = Image.open('./temp/{}.jpg'.format(EXP_ID))
    rgb_arr = pad_im(rgb_im)
    plt.close('all')
    return rgb_arr

def draw_graph_with_types(nds, tps):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(nds):
        _type = ROOM_CLASS[label]
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in tps:
        if m > 0:
            G_true.add_edges_from([(k, l)], color='b',weight=4)    
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u,v in edges]
    weights = [4 for u, v in edges]

    print(colors_H)
    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=16, font_color='white', font_weight='bold', edges=edges, edge_color=colors, width=weights, with_labels=True)
    plt.tight_layout()
    plt.savefig('./temp/{}.png'.format(EXP_ID), format="png")
    rgb_im = Image.open('./temp/{}.png'.format(EXP_ID))
    rgb_arr = pad_im(rgb_im)
    plt.close('all')
    return np.array(rgb_arr)

def bb_to_img(bbs, graphs, room_to_sample, triple_to_sample, boundary_bb=None, max_num_nodes=10, im_size=512, disc_scores=None):
	imgs = []
	nodes, triples = graphs
	bbs = np.array(bbs)
	for k in range(bbs.shape[0]):
		
		# Draw graph image
		inds = torch.nonzero(triple_to_sample == k)[:, 0]
		tps = triples[inds]
		inds = torch.nonzero(room_to_sample == k)[:, 0]
		offset = torch.min(inds)
		nds = nodes[inds]
		
		s, p, o = tps.chunk(3, dim=1)          
		s, p, o = [x.squeeze(1) for x in [s, p, o]] 
		eds = torch.stack([s, o], dim=1)          
		eds = eds-offset

		# Draw BB image
		bb = bbs[k]
		nds = nodes.view(-1, max_num_nodes)[k, :].detach().cpu().numpy()
		image_arr = np.zeros((im_size, im_size))
		im = Image.fromarray(image_arr.astype('uint8')).convert('RGB')
		dr = ImageDraw.Draw(im)

		for box, node in zip(bb, nds):
			node = node+1
			x0, y0, x1, y1 = box
			color = ID_COLOR[node]
			dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline=color)

		image_tensor = torch.tensor(np.array(im).transpose(2, 0, 1)/255.0)
		imgs.append(image_tensor)

	imgs_tensor = torch.stack(imgs)

	return imgs_tensor

def mask_to_bb(mask):
    
    # get masks pixels
    inds = np.array(np.where(mask>0))
    
    if inds.shape[-1] == 0:
        return [0, 0, 0, 0]

    # Compute BBs
    y0, x0 = np.min(inds, -1)
    y1, x1 = np.max(inds, -1)

    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, 255), min(x1, 255)

    w = x1 - x0
    h = y1 - y0
    x, y = x0, y0
    
    return [x0, y0, x1+1, y1+1]

def extract_corners(bb1, bb2, im_size=256):

	# initialize
	corners_set = set()
	x0, y0, x1, y1 = bb1
	x2, y2, x3, y3 = bb2

	# add corners from bbs
	corners_set.add((int(x0*im_size), int(y0*im_size)))
	corners_set.add((int(x0*im_size), int(y1*im_size)))
	corners_set.add((int(x1*im_size), int(y0*im_size)))
	corners_set.add((int(x1*im_size), int(y1*im_size)))
	corners_set.add((int(x2*im_size), int(y2*im_size)))
	corners_set.add((int(x2*im_size), int(y3*im_size)))
	corners_set.add((int(x3*im_size), int(y2*im_size)))
	corners_set.add((int(x3*im_size), int(y3*im_size)))

	# add intersection corners
	es1 = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
	es2 = [(x2, y2, x3, y2), (x3, y2, x3, y3), (x3, y3, x2, y3), (x2, y3, x2, y2)]

	for e1 in es1:
		for e2 in es2:
			x0, y0, x1, y1 = e1
			x2, y2, x3, y3 = e2

			e1_im = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(e1_im)
			dr.line((x0*im_size, y0*im_size, x1*im_size, y1*im_size), fill='white', width=1)
			e1_im = np.array(e1_im)/255.0

			e2_im = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(e2_im)
			dr.line((x2*im_size, y2*im_size, x3*im_size, y3*im_size), fill='white', width=1)
			e2_im = np.array(e2_im)/255.0

			cs_inter = np.array(np.where(e1_im + e2_im > 1))
			if(cs_inter.shape[1] == 1):
				corners_set.add((cs_inter[1][0], cs_inter[0][0]))

	return corners_set


def align_bb(bbs_batch, th=0.03):
	new_bbs_batch = bbs_batch.copy()
# 	np.save('debug.npy', new_bbs_batch)
# 	new_bbs_batch = np.load('debug.npy')
	for bbs in new_bbs_batch:
		## DEBUG
# 		im_deb1 = Image.new('RGB', (256, 256))
# 		dr = ImageDraw.Draw(im_deb1)
# 		for i, bb in enumerate(bbs):
# 			x0, y0, x1, y1 = bb * 255.0
# 			if i != 6:
# 				dr.rectangle((x0, y0, x1, y1), outline='green')
# 			else:
# 				dr.rectangle((x0, y0, x1, y1), outline='white')
		## DEBUG

		for i, bb1 in enumerate(bbs):
			x0, y0, x1, y1 = bb1
			x0_avg, y0_avg, x1_avg, y1_avg = [], [], [], []
			tracker = []
			for j, bb2 in enumerate(bbs):
				x2, y2, x3, y3 = bb2
				# horizontals
				if abs(x2-x0) <= th:
					x0_avg.append(x2) 
					tracker.append((j, 0, 0))
				if abs(x3-x0) <= th:
					x0_avg.append(x3)
					tracker.append((j, 2, 0))
				if abs(x2-x1) <= th:
					x1_avg.append(x2)
					tracker.append((j, 0, 2))
				if abs(x3-x1) <= th:
					x1_avg.append(x3)
					tracker.append((j, 2, 2))
				# verticals
				if abs(y2-y0) <= th:
					y0_avg.append(y2)
					tracker.append((j, 1, 1))
				if abs(y3-y0) <= th:
					y0_avg.append(y3)
					tracker.append((j, 3, 1))
				if abs(y2-y1) <= th:
					y1_avg.append(y2)
					tracker.append((j, 1, 3))
				if abs(y3-y1) <= th:
					y1_avg.append(y3)
					tracker.append((j, 3, 3))
			avg_vec = [np.mean(x0_avg), np.mean(y0_avg), np.mean(x1_avg), np.mean(y1_avg)]
			for l, val in enumerate(avg_vec):
				if not np.isnan(avg_vec[l]):
					bbs[i, l] = avg_vec[l]
			for k, l, m in tracker:
				if not np.isnan(avg_vec[m]):
					bbs[k, l] = avg_vec[m]

# 		## DEBUG
# 		im_deb2 = Image.new('RGB', (256, 256))
# 		dr = ImageDraw.Draw(im_deb2)
# 		for bb in bbs:
# 			x0, y0, x1, y1 = bb * 255.0
# 			dr.rectangle((x0, y0, x1, y1), outline='red')
# 		## DEBUG
# 		im_deb1.save('deb_1.jpg')
# 		im_deb2.save('deb_2.jpg')
	return new_bbs_batch

def remove_junctions(junctions, juncs_on, lines_on, delta=10.0):

    curr_juncs_on, curr_lines_on = list(juncs_on), list(lines_on)
    while True:
        new_lines_on, new_juncs_on = [], []
        is_mod = False
        for j1 in curr_juncs_on:
            adj_js, adj_as, ls = [], [], []
            for j2 in curr_juncs_on:
                if ((j1, j2) in curr_lines_on) or ((j2, j1) in curr_lines_on):
                    adj_js.append(j2)
                    pt1 = junctions[j1]
                    pt2 = junctions[j2]
                    adj_as.append(getAngle(pt1, pt2))
                    ls.append((j1, j2))

            if len(adj_js) > 2 or is_mod or len(adj_js) == 1:
                new_juncs_on.append(j1)
                new_lines_on += ls
            elif len(adj_js) == 2:
                diff = np.abs(180.0-np.abs(adj_as[0]-adj_as[1]))
                if diff >= delta:
                    new_juncs_on.append(j1)
                    new_lines_on += ls
                else:
                    new_lines_on.append((adj_js[0], adj_js[1]))
                    is_mod = True
        curr_juncs_on, curr_lines_on = list(new_juncs_on), list(new_lines_on)
        if not is_mod:
            break

    return curr_juncs_on, curr_lines_on

def bb_to_seg(bbs_batch, im_size=256):

	all_rooms_batch = []
	for bbs in bbs_batch:
		areas = np.array([(x1-x0)*(y1-y0) for x0, y0, x1, y1 in bbs])
		inds = np.argsort(areas)[::-1]
		bbs = bbs[inds]
		tag = 1
		rooms_im = np.zeros((256, 256))

		for (x0, y0, x1, y1) in bbs:
			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
				continue
			else:
				room_im = Image.new('L', (256, 256))
				dr = ImageDraw.Draw(room_im)
				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white', fill='white')
				inds = np.array(np.where(np.array(room_im) > 0))
				rooms_im[inds[1, :], inds[0, :]] = tag
				tag += 1

		all_rooms = []
		for tag in range(1, bbs.shape[0]+1):
			room = np.zeros((256, 256))
			inds = np.array(np.where(rooms_im == tag))
			room[inds[0, :], inds[1, :]] = 1.0
			all_rooms.append(room)
		all_rooms_batch.append(all_rooms)
	all_rooms_batch = np.array(all_rooms_batch)

# 	edges_batch = []
# 	for b in range(all_rooms_batch.shape[0]):
# 		edge_arr = []
# 		for k in range(all_rooms_batch.shape[1]):
# 			rm_arr = all_rooms_batch[b, k, :, :]
# 			rm_im = Image.fromarray(rm_arr*255)
# 			rm_im_lg = rm_im.filter(ImageFilter.MaxFilter(5))
# 			rm_im_sm = rm_im.filter(ImageFilter.MinFilter(5))
# 			edge_arr.append(np.array(rm_im_lg) - np.array(rm_im_sm))
# 		edges_batch.append(edge_arr)
# 	edges_batch = np.array(edges_batch)
# 	print(edges_batch.shape)

	return all_rooms_batch

def bb_to_im_fid(bbs_batch, nodes, im_size=299):
  nodes = np.array(nodes)
  bbs = np.array(bbs_batch[0])
  areas = np.array([(x1-x0)*(y1-y0) for x0, y0, x1, y1 in bbs])
  inds = np.argsort(areas)[::-1]
  bbs = bbs[inds]
  nodes = nodes[inds]
  im = Image.new('RGB', (im_size, im_size), 'white')
  dr = ImageDraw.Draw(im)
  for (x0, y0, x1, y1), nd in zip(bbs, nodes):
      if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
          continue
      else:
          color = ID_COLOR[int(nd)+1]
          dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), width=3, outline='black', fill=color)
  return im

# def bb_to_seg(bbs_batch, im_size=256, num_bbs=10):

# 	all_rooms_batch = []
# 	for bbs in bbs_batch:
# 		bbs = bbs.reshape(num_bbs, 4)
# 		inds = list(range(num_bbs))
# 		random.shuffle(inds)
# 		bbs = bbs[inds]
# 		tag = 1
# 		rooms_im = np.zeros((256, 256))

# 		for (x0, y0, x1, y1) in bbs:
# 			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# 				continue
# 			else:
# 				room_im = Image.new('L', (256, 256))
# 				dr = ImageDraw.Draw(room_im)
# 				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white', fill='white')
# 				inds = np.array(np.where(np.array(room_im) > 0))
# 				rooms_im[inds[0, :], inds[1, :]] = tag
# 				tag += 1

# 		all_rooms = []
# 		for tag in range(1, bbs.shape[0]+1):
# 			room = np.zeros((256, 256))
# 			inds = np.array(np.where(rooms_im == tag))
# 			room[inds[0, :], inds[1, :]] = 1.0
# 			all_rooms.append(room)
# 		all_rooms_batch.append(all_rooms)
# 	all_rooms_batch = np.array(all_rooms_batch)

# 	all_rooms_sm_batch = []
# 	edges_batch = []
# 	for b in range(all_rooms_batch.shape[0]):
# 		edge_arr = np.zeros((256, 256))
# 		all_rooms_sm = []
# 		for k in range(all_rooms_batch.shape[1]):
# 			rm_arr = all_rooms_batch[b, k, :, :]
# 			rm_im = Image.fromarray(rm_arr*255)
# 			rm_im_lg = rm_im.filter(ImageFilter.MaxFilter(5))
# 			rm_im_sm = rm_im.filter(ImageFilter.MinFilter(5))
# 			all_rooms_sm.append(np.array(rm_im_sm)/255.0)
# 			edge_arr += np.array(rm_im_lg) - np.array(rm_im_sm)
# 		edge_arr = np.clip(edge_arr, 0, 255)
# 		edges_batch.append(edge_arr/255.0)
# 		all_rooms_sm_batch.append(all_rooms_sm)
# 	edges_batch = np.array(edges_batch)[:, np.newaxis, :, :]

# 	all_rooms_sm_batch = np.array(all_rooms_sm_batch)
# 	all_rooms_sm_batch = np.sum(all_rooms_sm_batch, 1)[:, np.newaxis, :, :]
# 	all_rooms_sm_batch = np.clip(all_rooms_sm_batch, 0, 1)
# 	edges_batch = np.concatenate([np.zeros((all_rooms_sm_batch.shape[0], 3, 256, 256)), all_rooms_sm_batch, np.zeros((all_rooms_sm_batch.shape[0], 7, 256, 256)), edges_batch], 1)

# # 	edges_batch = np.concatenate([np.zeros((all_rooms_batch.shape[0], 1, 256, 256)), edges_batch], 1)
# # 	all_rooms_batch = np.concatenate([all_rooms_batch, edges_batch], 1)

# 	return edges_batch

def get_type(pxs):
	ori_arr = [0, 0, 0, 0]
	for p in pxs:
		if tuple(p) == (0, 1):
			ori_arr[0] = 1
		if tuple(p) == (1, 2):
			ori_arr[1] = 1
		if tuple(p) == (2, 1):
			ori_arr[2] = 1
		if tuple(p) == (1, 0):
			ori_arr[3] = 1

# POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )], [(0, 3), (0, 1), (1, 2), (2, 3)], [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], [(0, 1, 2, 3)]]
	# Type 1
	if tuple(ori_arr) == tuple([0, 0, 1, 0]):
		return 0
	# Type 2
	if tuple(ori_arr) == tuple([0, 0, 0, 1]):
		return 1
	# Type 3
	if tuple(ori_arr) == tuple([1, 0, 0, 0]):
		return 2
	# Type 4
	if tuple(ori_arr) == tuple([0, 1, 0, 0]):
		return 3
	# Type 5
	if tuple(ori_arr) == tuple([1, 0, 0, 1]):
		return 4
	# Type 6
	if tuple(ori_arr) == tuple([1, 1, 0, 0]):
		return 5
	# Type 7
	if tuple(ori_arr) == tuple([0, 1, 1, 0]):
		return 6
	# Type 8
	if tuple(ori_arr) == tuple([0, 0, 1, 1]):
		return 7
	# Type 9
	if tuple(ori_arr) == tuple([0, 1, 1, 1]):
		return 8
	# Type 10
	if tuple(ori_arr) == tuple([1, 0, 1, 1]):
		return 9
	# Type 11
	if tuple(ori_arr) == tuple([1, 1, 0, 1]):
		return 10
	# Type 12
	if tuple(ori_arr) == tuple([1, 1, 1, 0]):
		return 11
	# Type 13
	if tuple(ori_arr) == tuple([1, 1, 1, 1]):
		return 12

def bb_to_vec(bbs_batch, im_size=256):
	cs_type_batch = []
	for bbs in bbs_batch:
		corners_set = set()
		for x0, y0, x1, y1 in bbs:
			x0, y0, x1, y1 = int(x0*255.0), int(y0*255.0), int(x1*255.0), int(y1*255.0)
			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
				continue
			else:
				corners_set.add((x0, y0))
				corners_set.add((x0, y1))
				corners_set.add((x1, y0))
				corners_set.add((x1, y1))
# 		corners_set_aug = set()
# 		for x0, y0 in list(corners_set):
# 			for x1, y1 in list(corners_set):
# 				corners_set_aug.add((x0, y0))
# 				corners_set_aug.add((x0, y1))
# 				corners_set_aug.add((x1, y0))
# 				corners_set_aug.add((x1, y1))
		cs_type_batch.append(list(corners_set))
	return cs_type_batch

# def bb_to_vec(bbs_batch, im_size=256, num_bbs=10):
# 	bbs_batch = bbs_batch.detach().cpu().numpy()
# 	cs_type_batch = []
# 	cs_batch = []
# 	for bbs in bbs_batch:
# 		bbs = bbs.reshape(num_bbs, 4)
# 		corners_set = set()
# 		for (x0, y0, x1, y1) in bbs:
# 			for (x2, y2, x3, y3) in bbs:
# 				bb1 = (x0, y0, x1, y1)
# 				bb2 = (x2, y2, x3, y3)
# 				if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# 					continue
# 				elif x2 < 0 or y2 < 0 or x3 < 0 or y3 < 0:
# 					continue
# 				else:
# 					corners_set = corners_set.union(extract_corners(bb1, bb2))

# 		bbs_im = Image.new('L', (256, 256))
# 		dr = ImageDraw.Draw(bbs_im)
# 		for (x0, y0, x1, y1) in bbs:
# 			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# 				continue
# 			else:
# 				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white')

# # 		bbs_im.save('./debug0.jpg')
# 		bbs_im = np.array(bbs_im)
# 		corners_set = np.array(list(corners_set))
# 		cs_type_sample = [[], [], [], [], [], [], [], [], [], [], [], [], []]
# 		for c in corners_set:
# 			y, x = c
# 			c_im = np.zeros((256, 256))
# 			c_im[x, y] = 255
# 			c_im = Image.fromarray(c_im.astype('uint8'))
# # 			print(x-1, x+2, y-1, y+2)
# 			pxs = np.array(np.where(bbs_im[x-1:x+2, y-1:y+2] > 0)).transpose()
# 			if(pxs.shape[0] == 0):
# 				print(bbs_im[x-1:x+2, y-1:y+2])
# 				print(x, y)
				
# 			_type = get_type(pxs)
# 			if _type is not None:
# 				cs_type_sample[_type].append((x, y))
# 		cs_type_batch.append(cs_type_sample)

# # 		# debug
# # 		bbs_im_debug = Image.new('L', (256, 256))
# # 		dr = ImageDraw.Draw(bbs_im_debug)
# # 		for (x0, y0, x1, y1) in bbs:
# # 			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# # 				continue
# # 			else:
# # 				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white')
# # 		bbs_im_debug.save('./debug1.jpg')

# # 		corners_d ebug = Image.new('L', (256, 256))
# # 		dr = ImageDraw.Draw(corners_debug)
# # 		for x, y in list(corners_set):
# # 			dr.ellipse((x-2, y-2, x+2, y+2), outline='white')
# # 		corners_debug.save('./debug2.jpg')
# # 		print(corners_set)
# # 		print(np.array(list(corners_set)).shape)

# 	return cs_type_batch

def  visualizeCorners(wallPoints):
	im_deb = Image.new('RGB', (256, 256))
	dr = ImageDraw.Draw(im_deb)
	for (x, y, i, j) in wallPoints:
		dr.ellipse((x-1, y-1, x+1, y+1), fill='red')
		font = ImageFont.truetype("arial.ttf", 10)
		dr.text((x, y), str(3*i+j+1),(255,255,255), font=font)
	im_deb.save('./debug_all_corner_with_text.jpg')
	return

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
def rectangle_renderer(theta, im_size=64):
    
    # scale theta
    theta = theta*im_size
    
    # create meshgrid
    xs = np.arange(im_size)
    ys = np.arange(im_size)
    xs, ys = np.meshgrid(xs, ys)
    xs = torch.tensor(np.repeat(xs[np.newaxis, :, :], theta.shape[0], axis=0)).float().cuda()
    ys = torch.tensor(np.repeat(ys[np.newaxis, :, :], theta.shape[0], axis=0)).float().cuda()

    # conditions
    cond_1 = torch.min(torch.cat([F.relu(ys - theta[:, 1].view(-1, 1, 1)).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0] * \
             torch.min(torch.cat([F.relu(theta[:, 3].view(-1, 1, 1) - ys).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0]
            
    cond_2 = torch.min(torch.cat([F.relu(xs - theta[:, 0].view(-1, 1, 1)).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0] * \
             torch.min(torch.cat([F.relu(theta[:, 2].view(-1, 1, 1) - xs).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0]

    # lines
    line_1 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(xs - torch.ones((theta.shape[0], im_size, im_size)).cuda() - theta[:, 0].view(-1, 1, 1))) * cond_1).view(-1, im_size, im_size, 1)    # top
    line_2 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(xs + torch.ones((theta.shape[0], im_size, im_size)).cuda() - theta[:, 2].view(-1, 1, 1))) * cond_1).view(-1, im_size, im_size, 1)    # bottom
    line_3 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(ys - theta[:, 1].view(-1, 1, 1))) * cond_2).view(-1, im_size, im_size, 1)        # left
    line_4 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(ys - theta[:, 3].view(-1, 1, 1))) * cond_2).view(-1, im_size, im_size, 1)        # right
            
    I = torch.max(torch.cat([line_1, line_2, line_3, line_4], -1), -1)[0]
    
    return I

def checkpoint(real_room_bb, fake_room_bb,  nodes, triples, room_to_sample, triple_to_sample, generator, exp_folder, batches_done, fake_validity, real_validity, boundary_bb, Tensor, latent_dim, out_imsize):
    
    torch.save(generator.state_dict(), './checkpoints/gen_neighbour_{}_{}.pth'.format(exp_folder, batches_done))
    fake_imgs_tensor = bb_to_img(fake_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample, \
                                 boundary_bb, disc_scores=fake_validity, im_size=out_imsize)
    real_imgs_tensor = bb_to_img(real_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample, \
                                 boundary_bb, disc_scores=real_validity, im_size=out_imsize)


    save_image(fake_imgs_tensor, "{}/fake_{}.png".format(exp_folder, batches_done), nrow=16)
    save_image(real_imgs_tensor, "{}/real_{}.png".format(exp_folder, batches_done), nrow=16)

    ## perform variation analysis
    # Sample noise as generator input
    layouts_imgs_tensor = []
    n_samples = 16
    for _ in range(10):

        # get partial batch
        z = Variable(Tensor(np.random.normal(0, 1, (real_room_bb.shape[0], latent_dim))))
        z_partial = z[:n_samples]
        nodes_partial = nodes[:n_samples*10]
        triples_partial = triples[:n_samples*45, :]
        room_to_sample_partial = room_to_sample[:n_samples*10]
        boundary_bb_partial = boundary_bb[:n_samples, :]
        triple_to_sample_partial = triple_to_sample[:n_samples*45]

        # plot images
        fake_room_bb_partial = generator(z_partial, [nodes_partial, triples_partial], room_to_sample_partial, boundary=boundary_bb_partial)
        fake_imgs_tensor = bb_to_img(fake_room_bb_partial.data, [nodes_partial, triples_partial], room_to_sample_partial, \
                                     triple_to_sample_partial, boundary_bb_partial, im_size=out_imsize)

        layouts_imgs_tensor.append(fake_imgs_tensor)
    layouts_imgs_tensor = torch.stack(layouts_imgs_tensor)
    layouts_imgs_tensor = layouts_imgs_tensor.view(10, 16, 2, 3, out_imsize, out_imsize)
    layouts_imgs_tensor_filtered = []
    for k in range(16):
        for l in range(10):
            if l == 0:
                layouts_imgs_tensor_filtered.append(layouts_imgs_tensor[l, k, 0, :, :, :])
            layouts_imgs_tensor_filtered.append(layouts_imgs_tensor[l, k, 1, :, :, :])
    layouts_imgs_tensor_filtered = torch.stack(layouts_imgs_tensor_filtered).contiguous().view(-1, 3, out_imsize, out_imsize)
    save_image(layouts_imgs_tensor_filtered, "{}/layouts_{}.png".format(exp_folder, batches_done), nrow=11)

# def combine_images(layout_batch, im_size=256):
#     layout_batch = layout_batch.detach().cpu().numpy()
#     all_imgs = []
#     for layout in layout_batch:
#         comb_img = Image.new('RGB', (im_size, im_size))
#         dr = ImageDraw.Draw(comb_img)
#         for i in range(layout.shape[1]):
#             for j in range(layout.shape[2]):
#                 h, w = layout[0, i, j], layout[1, i, j]
#                 if layout[2, i, j] > 0.5:
#                     label = 1 #np.argmax(layout[:10, i, j]) + 1
#                     h, w = layout[0, i, j], layout[1, i, j]
#                     color = ID_COLOR[int(label)]
#                     r = im_size/layout.shape[1]
#                     dr.rectangle((r*i-(im_size*h)/2.0, r*j-(im_size*w)/2.0, \
#                                   r*i+(im_size*h)/2.0, r*j+(im_size*w)/2.0), outline=color)
#         all_imgs.append(torch.tensor(np.array(comb_img).\
#                                      astype('float').\
#                                      transpose(2, 0, 1))/255.0)
#     all_imgs = torch.stack(all_imgs)
#     return all_imgs
            

def combine_images_bbs(bbs_batch, im_size=256):
    bbs_batch = bbs_batch.view(-1, 10, 4).detach().cpu().numpy()
    all_imgs = []
    for bbs in bbs_batch:
        comb_img = Image.new('RGB', (im_size, im_size))
        dr = ImageDraw.Draw(comb_img)
        for bb in bbs:
            x0, y0, x1, y1 = im_size*bb
            h = x1-x0
            w = y1-y0
            if h > 4 and w > 4:
                color = ID_COLOR[1]
                dr.rectangle((x0, y0, x1, y1), outline=color)
        all_imgs.append(torch.tensor(np.array(comb_img).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
    all_imgs = torch.stack(all_imgs)
    return all_imgs

def draw_masks(masks, real_nodes, im_size=256):
    real_nodes = np.where(real_nodes.copy()==1)[-1]
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 256))  # Semitransparent background.
    for m, nd in zip(masks, real_nodes):
        
        # draw region
        reg = Image.new('RGBA', (m.shape[0], m.shape[0]), (0,0,0,0))
        dr_reg = ImageDraw.Draw(reg)
        m[m>0] = 255
        m[m<0] = 0
        m = Image.fromarray(m)
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 32))
        reg = reg.resize((256, 256))
        bg_img.paste(Image.alpha_composite(bg_img, reg))
  
    for m, nd in zip(masks, real_nodes):
        cnt = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_cnt = ImageDraw.Draw(cnt)
        
        mask = np.zeros((256,256,3)).astype('uint8')
        m[m>0] = 255
        m[m<0] = 0
        m = m[:, :, np.newaxis].astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
        ret,thresh = cv2.threshold(m,127,255,0)
        contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
        if len(contours) > 0:  
            contours = [c for c in contours]
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)
        
        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert('L'), fill=(r, g, b, 256))
        
        bg_img.paste(Image.alpha_composite(bg_img, cnt))
    return bg_img.resize((im_size, im_size))


def extract_rooms(masks, real_nodes, im_size=256):
    real_nodes = real_nodes.copy()    
    types = []
    polys = []
    for m, nd in zip(masks, real_nodes):
        m = m.detach().cpu().numpy()
        m[m>0] = 255
        m[m<0] = 0
        m = m[:, :, np.newaxis].astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
        ret, thresh = cv2.threshold(m,127,255,0)
        contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
          contours = [c for c in contours]
        else:
          contours = []
        types.append(CLASS_ROM[nd+1])
        polys.append(contours)

    return types, polys

import webcolors

def combine_images_maps(maps_batch, nodes_batch, edges_batch, \
                        nd_to_sample, ed_to_sample, im_size=256):
    maps_batch = maps_batch.detach().cpu().numpy()
    nodes_batch = nodes_batch.detach().cpu().numpy()
    edges_batch = edges_batch.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    all_imgs = []
    shift = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b)
        inds_ed = np.where(ed_to_sample==b)
        
        mks = maps_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        eds = edges_batch[inds_ed]
        
        comb_img = np.ones((im_size, im_size, 3)) * 255
        extracted_rooms = []
        for mk, nd in zip(mks, nds):
            r =  im_size/mk.shape[-1]
            x0, y0, x1, y1 = np.array(mask_to_bb(mk)) * r 
            h = x1-x0
            w = y1-y0
            if h > 0 and w > 0:
                extracted_rooms.append([mk, (x0, y0, x1, y1), nd])
        
        # draw graph
        graph_img = draw_graph(nds, eds, shift)
        shift += len(nds)
        all_imgs.append(torch.FloatTensor(np.array(graph_img.convert('RGBA')).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
        
        # draw masks
        mks = np.array([m for m, _, _ in extracted_rooms])
        nds = np.array([n for _, _, n in extracted_rooms])
        comb_img = draw_masks(mks, nds)

        # mask_img = np.ones((32, 32, 3)) * 255
        # for rm in extracted_rooms:
        #     mk, _, nd = rm 
        #     inds = np.array(np.where(mk>0))
        #     _type = np.where(nd==1)[0]
        #     if len(_type) > 0:
        #         color = ID_COLOR[_type[0] + 1]
        #     else:
        #         color = 'black'
        #     r, g, b = webcolors.name_to_rgb(color)
        #     mask_img[inds[0, :], inds[1, :], :] = [r, g, b]
        # mask_img = Image.fromarray(mask_img.astype('uint8'))
        # mask_img = mask_img.resize((im_size, im_size))
        # all_imgs.append(torch.FloatTensor(np.array(mask_img).transpose(2, 0, 1))/255.0)
            
        # # draw boxes - filling
        # comb_img = Image.fromarray(comb_img.astype('uint8'))
        # dr = ImageDraw.Draw(comb_img)
        # for rm in extracted_rooms:
        #     _, rec, nd = rm 
        #     dr.rectangle(tuple(rec), fill='beige')
            
        # # draw boxes - outline
        # for rm in extracted_rooms:
        #     _, rec, nd = rm 
        #     _type = np.where(nd==1)[0]
        #     if len(_type) > 0:
        #         color = ID_COLOR[_type[0] + 1]
        #     else:
        #         color = 'black'
        #     dr.rectangle(tuple(rec), outline=color, width=4)
            
#         comb_img = comb_img.resize((im_size, im_size))
        all_imgs.append(torch.FloatTensor(np.array(comb_img).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
    all_imgs = torch.stack(all_imgs)
    return all_imgs


def getAngle(pt1, pt2):
    # return angle in clockwise direction
    x, y = pt1
    xn, yn = pt2
    dx, dy = xn-x, yn-y
    dir_x, dir_y = (dx, dy)/(np.linalg.norm([dx, dy])+1e-8)
    rad = np.arctan2(-dir_y, dir_x)
    ang = np.degrees(rad)
    if ang < 0:
        ang = (ang + 360) % 360
    return 360-ang

def preprocess(polys, ths=[2, 4]):

    #snap
    new_polys = []
    for p in polys:
      cs, es = p
      new_cs = np.array(cs)
      for th in ths:
          for i in range(len(new_cs)):
              x0, y0 = new_cs[i]
              x0_avg, y0_avg = [], []
              tracker = []
              for j in range(len(new_cs)):
                  x1, y1 = new_cs[j]

                  # horizontals
                  if abs(x1-x0) <= th:
                      x0_avg.append(x1) 
                      tracker.append((j, 0))
                  # verticals
                  if abs(y1-y0) <= th:
                      y0_avg.append(y1)
                      tracker.append((j, 1))
              avg_vec = [np.mean(x0_avg), np.mean(y0_avg)]

              # set others
              for k, m in tracker:
                  new_cs[k, m] = avg_vec[m]

      #supress
      final_cs = []
      final_es = []
      new_cs = np.array(new_cs)
      for x0, y0 in new_cs:
        in_set = False
        for x1, y1 in final_cs:
          if abs(x1-x0) <= 2 and abs(y1-y0) <= 2:
            in_set = True
            break
        if in_set == False:
          final_cs.append((x0, y0))

      # must be at least 4 corners
      if len(final_cs) >= 4:
        for k in range(len(final_cs)-1):
          final_es.append((k, k+1))
        final_es.append((len(final_cs)-1, 0))

        # remove colinear
        new_final_cs = []
        new_final_es = []
        for k in range(len(final_es)-1):
          e1 = final_es[k]
          e2 = final_es[k+1]
          pt1, pt2, pt3 = np.array(final_cs[e1[0]]), np.array(final_cs[e1[1]]), np.array(final_cs[e2[1]])
          a1 = getAngle(pt1, pt2)
          a2 = getAngle(pt2, pt3)
          if abs(a1-a2)%360 == 0:
            print('ERR NOT SUPPORTED', abs(a1-a2)%360, a1, a2)
        e1 = final_es[0]
        e2 = final_es[-1]
        pt1, pt2, pt3 = np.array(final_cs[e1[0]]), np.array(final_cs[e1[1]]), np.array(final_cs[e2[1]])
        a1 = getAngle(pt1, pt2)
        a2 = getAngle(pt2, pt3)
        if abs(a1-a2)%360 == 0:
          print('ERR NOT SUPPORTED', abs(a1-a2)%360, a1, a2)
        new_polys.append([final_cs, final_es])
      else: 
        new_polys.append([[], []])
  
    return new_polys

def extract_edges(polys):
  poly_primitives = []
  for ps in polys:
    if len(ps) > 0:
      poly = ps[0]
      corners = []
      edges = []

      corners = np.array(poly[:, 0, :])
      new_corners = []
      l = -1
      for k in range(len(corners)):
        xp, yp = corners[l]
        xc, yc = corners[k]
        if (abs(xp-xc) <= 2) and (abs(yp-yc) <= 2):
          continue
        elif abs(xp-xc) <= 2:
          xc = xp
        elif abs(yp-yc) <= 2:
          yc = yp
        else:
          print('ERR broken polygon')
        new_corners.append((xc, yc))
        l = k
      corners = new_corners

      for k in range(len(corners)-1):
        edges.append((k, k+1))
      edges.append((len(corners)-1, 0))
      poly_primitives.append([corners, edges])
    else:
      poly_primitives.append([[], []])
  return poly_primitives

def draw_vecs(polys, real_nodes, im_size=512):
    real_nodes = real_nodes.copy()
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 256))  # Semitransparent background.
    for p, nd in zip(polys, real_nodes):
      
        # draw region
        reg = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_reg = ImageDraw.Draw(reg)
        color = ID_COLOR[ROOM_CLASS[nd]]
        m = Image.new('L', (256, 256))
        dr_m = ImageDraw.Draw(m)
        cs, es = p
        if len(cs) > 2:
          cs = [(x, y) for x, y in cs]
          dr_m.polygon(cs, fill='white')
          r, g, b = webcolors.name_to_rgb(color)
          dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 32))
          reg = reg.resize((256, 256))
          bg_img.paste(Image.alpha_composite(bg_img, reg))

    for p, nd in zip(polys, real_nodes):
        cnt = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_cnt = ImageDraw.Draw(cnt)
        cs, es = p
        color = ID_COLOR[ROOM_CLASS[nd]]
        r, g, b = webcolors.name_to_rgb(color)
        m = Image.new('L', (256, 256))
        dr_m = ImageDraw.Draw(m)
        if len(cs) > 2:
          for e in es:
            x0, y0 = cs[e[0]]
            x1, y1 = cs[e[1]]
            dr_m.line((x0, y0, x1, y1), fill='white', width=2)
          dr_cnt.bitmap((0, 0), m, fill=(r, g, b, 228))
          bg_img.paste(Image.alpha_composite(bg_img, cnt))

    return bg_img.resize((im_size, im_size))

def _convert_poly_to_mask(polys):
  masks = [] 
  for p in polys:
    m = Image.new('L', (256, 256))
    dr_m = ImageDraw.Draw(m)
    cs, es = p
    if len(cs) > 2:
      cs = [(x, y) for x, y in cs]
      dr_m.polygon(cs, fill='white')
    masks.append(np.array(m))
  return np.array(masks)

def visualize_sample(types, polys):
  return np.array(draw_vecs(polys, types))

def visualize_vector(types, masks, corners, edges):
  return

def _project_and_trace(masks):
  areas = [(m/255.0).sum() for m in masks]
  inds = sorted(range(len(areas)),key=areas.__getitem__)[::-1]
  floorplan = np.zeros_like(masks[0])
  tag = 1
  ind_to_tag = {}

  # project all rooms
  for i in inds:
    floorplan[np.where(masks[i] > 0)] = tag
    ind_to_tag[i] = tag
    tag+=1
    
  # extract new masks
  final_cnts = []
  for i, m in enumerate(masks):
    new_m = np.zeros_like(m)
    tag = ind_to_tag[i]
    new_m[np.where(floorplan == tag)] = 255.0
    ret, thresh = cv2.threshold(new_m,127,255,0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:  

      # keep only largest contours
      cnt_masks = []
      cnt_coords = []
      for c in contours:
        c_m = Image.new('L', (256, 256))
        d = ImageDraw.Draw(c_m)
        c = c[:, 0, :]
        poly = [(x, y) for x, y in c]
        d.polygon(poly, fill='white')
        cnt_coords.append(poly)
        cnt_masks.append(np.array(c_m))
      areas = [(m/255.0).sum() for m in cnt_masks]
      ind = sorted(range(len(areas)),key=areas.__getitem__)[-1]
      final_cnts.append(cnt_coords[ind])
    else:
      final_cnts.append([])

  # # debug
  # for m, cnt in zip(masks, final_cnts):
  #   c_m = Image.new('L', (256, 256))
  #   d = ImageDraw.Draw(c_m)
  #   if len(cnt) > 2:
  #     d.polygon(cnt, fill='white')
  #   print(cnt)
  #   plt.figure()
  #   plt.imshow(Image.fromarray(m.astype('uint8')))
  #   plt.figure()
  #   plt.imshow(c_m)
  #   plt.show()
  return final_cnts


def _extract_corners_and_edges(cnts):
  graph = defaultdict(list)
  for cnt in cnts:
    if len(cnt) > 2:
      x_prev, y_prev = cnt[-1]
      for (x_curr, y_curr) in cnt:
        graph[(x_curr, y_curr)].append((x_prev, y_prev))
        graph[(x_prev, y_prev)].append((x_curr, y_curr))
        x_prev, y_prev = x_curr, y_curr
  return graph

def _suppress_corners(graph, theshold=4):
  while True:
    new_graph = defaultdict(list)
    _found_close_corners = False
    _c_x = None
    _c_y = None

    # look for corners
    for k, c1 in enumerate(graph.keys()):
      for l, c2 in enumerate(graph.keys()):
        dist = np.linalg.norm(np.array(c1)-np.array(c2))
        if (k > l) and (dist <= theshold):
          _found_close_corners = True
          _c_x, _c_y = c1, c2
          break
      if _found_close_corners:
        break

    # merge corners
    if _found_close_corners:
      _c_z = ((_c_x[0]+_c_y[0])/2.0, (_c_x[1]+_c_y[1])/2.0)
      for c in graph[_c_x]:
        if c not in new_graph[_c_z]:
          new_graph[_c_z].append(c)
      for c in graph[_c_y]:
        if c not in new_graph[_c_z]:
          new_graph[_c_z].append(c)

      for c1 in graph.keys():
        if (c1 != _c_x) and (c1 != _c_y):
          for c2 in graph[c1]:
            if (c2 == _c_x) or (c2 == _c_y):
              new_graph[c1].append(_c_z)
            else:
              new_graph[c1].append(c2)
      graph = new_graph
    else:
      break
  return graph

def _snap_corners(graph, ths=[2, 4]):

  # store edges
  edges = []
  for k, c1 in enumerate(graph.keys()):
    for l, c2 in enumerate(graph.keys()):
      if (k > l) and (c2 in graph[c1]):
        edges.append((k, l))

  new_cs = np.array(list(graph.keys()))
  for th in ths:
      for i in range(len(new_cs)):
          x0, y0 = new_cs[i]
          x0_avg, y0_avg = [], []
          tracker = []
          for j in range(len(new_cs)):
              x1, y1 = new_cs[j]

              # horizontals
              if abs(x1-x0) <= th:
                  x0_avg.append(x1) 
                  tracker.append((j, 0))
              # verticals
              if abs(y1-y0) <= th:
                  y0_avg.append(y1)
                  tracker.append((j, 1))
          avg_vec = [np.mean(x0_avg), np.mean(y0_avg)]

          # set others
          for k, m in tracker:
              new_cs[k, m] = avg_vec[m]

  # building new graph
  new_graph = defaultdict(list)
  for k, l in edges:
    tuple_k = tuple(new_cs[k])
    tuple_l = tuple(new_cs[l])
    new_graph[tuple_k].append(tuple_l)
    new_graph[tuple_l].append(tuple_k)

  return new_graph

def _flood_fill(edge_mask, x0, y0, tag):
    new_edge_mask = np.array(edge_mask)
    nodes = [(x0, y0)]
    new_edge_mask[x0, y0] = tag
    while len(nodes) > 0:
        x, y = nodes.pop(0)
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (0 <= x+dx < new_edge_mask.shape[0]) and (0 <= y+dy < new_edge_mask.shape[0]) and (new_edge_mask[x+dx, y+dy] == 0):
                new_edge_mask[x+dx, y+dy] = tag
                nodes.append((x+dx, y+dy))
    return new_edge_mask

def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

def compute_edges_mask(graph):
  edge_mask = Image.new('L', (256, 256)) 
  dr = ImageDraw.Draw(edge_mask)
  for c1 in graph.keys():
    for c2 in graph[c1]:
      x1, y1 = c1
      x2, y2 = c2
      dr.line((x1, y1, x2, y2), width=2, fill='white')
  return np.array(edge_mask)/255.0

def _update_room_masks(wrong_masks, graph):
  edge_mask = compute_edges_mask(graph)
  region_mask = fill_regions(edge_mask)
  masks, boxes, labels = [], [], []
  inds = np.where((region_mask > 2) & (region_mask < 255))
  tags = set(region_mask[inds])
  for t in tags:
      m = np.zeros((256, 256))
      inds = np.array(np.where(region_mask == t))
      m[inds[0, :], inds[1, :]] = 1.0
      masks.append(m)
  curr_masks = np.stack(masks)

  fixed_masks = []
  for m_w in wrong_masks:
    if np.array(m_w).sum() > 0:
      best_iou = -1
      best_ind = -1
      for k, m_c in enumerate(curr_masks):
        iou = np.logical_and(m_c, m_w).sum()/np.logical_or(m_c, m_w).sum()
        if iou > best_iou:
          best_iou = iou
          best_ind = k
      fixed_masks.append(curr_masks[best_ind])
    else:
      fixed_masks.append(np.zeros_like(m_w))

  return fixed_masks

def _visualize_floorplan_graph(graph, types, masks, im_size=512):

  dwg = svgwrite.Drawing('_temp.svg', (256, 256))
  drawn_edges = []
  for x1, y1 in graph.keys():
    for x2, y2 in graph[(x1, y1)]:
      if ((x1, y1, x2, y2) in drawn_edges) or ((x2, y2, x1, y1) in drawn_edges):
        continue
      dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=2, opacity=0.7))

  # draw edges
  for x, y in graph.keys():
    dwg.add(dwg.circle(center=(float(x), float(y)), r=1.5, stroke='red', fill='white', stroke_width=1, opacity=0.7))
      
  # draw regions
  reg = Image.new('RGBA', (256, 256), (0,0,0,0))
  dr_reg = ImageDraw.Draw(reg)
  for k, t, m in zip(range(len(types)), types, masks):
    color = ID_COLOR[ROOM_CLASS[t]]
    r, g, b = webcolors.name_to_rgb(color)
    y, x = np.array(np.where(m > 0)).mean(-1)
    m = Image.fromarray((m*255.0).astype('uint8'))
    m = m.filter(ImageFilter.MaxFilter(3))
    dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 64))
    # fnt = ImageFont.truetype('Pillow/Tests/fonts/Arial.ttf', 12)
    fnt = ImageFont.truetype("arial.ttf", 12)
    dr_reg.text((x, y), str(k), font=fnt, fill=(255,0,0,256))

  reg.save('_rooms_temp.png')
  dwg.add(svgwrite.image.Image(os.path.abspath('_rooms_temp.png'), size=(256, 256)))      
  dwg.save()
  
  print('running inkscape ...')
  os.system('inkscape ./_temp.svg --export-png=_temp.png -w {}'.format(im_size))
  png_im = Image.open("_temp.png")
  
  rgb_img = Image.new('RGBA', (im_size, im_size), 'white')
  rgb_img.paste(png_im, (0, 0), mask=png_im) 

  return rgb_img

def _extract_vector_format(cnts):
  fp_graph = _extract_corners_and_edges(cnts)
  fp_graph = _suppress_corners(fp_graph)
  fp_graph = _snap_corners(fp_graph)
  return fp_graph

def vectorize_heuristic(types, polys):

  # first get room masks and project rooms
  masks =_convert_poly_to_mask(polys)
  cnts = _project_and_trace(masks)

  # extract projected floorplan vector 
  fp_graph = _extract_vector_format(cnts)

  # visualize floorplan in vector format
  masks = _update_room_masks(masks, fp_graph)
  rgb_img = _visualize_floorplan_graph(fp_graph, types, masks)

  return np.array(rgb_img)

def check_polygon_intersection(p1, p2):
  cs1, es1 = p1
  cs2, es2 = p2

  r1 = Image.new('L', (256, 256))
  r2 = Image.new('L', (256, 256))
  dr1 = ImageDraw.Draw(r1)
  dr2 = ImageDraw.Draw(r2)

  if len(cs1) > 2 and len(cs2) > 2:
    cs1 = [(x, y) for x, y in cs1]  
    dr1.polygon(cs1, fill='white')

    cs2 = [(x, y) for x, y in cs2]  
    dr2.polygon(cs2, fill='white')

    r1_arr = np.array(r1)
    r2_arr = np.array(r2)

    if np.logical_and(r1_arr, r2_arr).sum() > 0:
      return True
  return False

def iou_polygon_intersection(p1, p2):
  cs1, es1 = p1
  cs2, es2 = p2

  r1 = Image.new('L', (256, 256))
  r2 = Image.new('L', (256, 256))
  dr1 = ImageDraw.Draw(r1)
  dr2 = ImageDraw.Draw(r2)

  if len(cs1) > 2 and len(cs2) > 2:
    cs1 = [(x, y) for x, y in cs1]  
    dr1.polygon(cs1, fill='white')

    cs2 = [(x, y) for x, y in cs2]  
    dr2.polygon(cs2, fill='white')

    r1_arr = np.array(r1)
    r2_arr = np.array(r2)

    return np.logical_and(r1_arr, r2_arr).sum()/np.logical_or(r1_arr, r2_arr).sum()
  return 0.0

def check_polygon_connectivity(p1, p2):
  cs1, es1 = p1
  cs2, es2 = p2
  for e1 in es1:
    for e2 in es2:
      x0, y0 = cs1[e1[0]]
      x1, y1 = cs1[e1[1]] 
      x2, y2 = cs2[e2[0]]
      x3, y3 = cs2[e2[1]]  

      dx1, dx2 = abs(x1-x0), abs(x3-x2)
      dy1, dy2 = abs(y1-y0), abs(y3-y2)

      if (dx1 < dy1) and (dx2 < dy2):
        y1_min, y2_min = min(y0, y1), min(y2, y3)
        y1_max, y2_max = max(y0, y1), max(y2, y3)
        dist_x = abs(x2-x0)
        c1 = y2_min < y1_min < y2_max
        c2 = y2_min < y1_max < y2_max
        c3 = y1_min < y2_min < y1_max
        c4 = y1_min < y2_max < y1_max
        if dist_x < 2 and (c1 or c2 or c3 or c4):
          return True

      elif (dx1 > dy1) and (dx2 > dy2):
        x1_min, x2_min = min(x0, x1), min(x2, x3)
        x1_max, x2_max = max(x0, x1), max(x2, x3)
        dist_y = abs(y2-y0)
        c1 = x2_min < x1_min < x2_max
        c2 = x2_min < x1_max < x2_max
        c3 = x1_min < x2_min < x1_max
        c4 = x1_min < x2_max < x1_max
        if dist_y < 2 and (c1 or c2 or c3 or c4):
          return True

  return  False

def split_edge(new_polys, p_ind):

  # randomly pick an edge
  # if random.uniform(0, 1) < 0.5:
  #   return new_polys

  new_polys = copy.deepcopy(new_polys)
  new_cs, new_es = new_polys[p_ind]
  k = np.random.choice(range(len(new_es)))
  e = new_es[k]
  x0, y0 = new_cs[e[0]]
  x1, y1 = new_cs[e[1]]

  # move
  split = random.uniform(0, 1) 
  break_pt = np.array(new_cs[e[0]]) + split*(np.array(new_cs[e[1]])-np.array(new_cs[e[0]]))
  break_pt = break_pt.astype('int')
  
  # update
  new_cs.insert(e[0]+1, tuple(break_pt))
  filtered_new_es = []
  for l in range(len(new_cs)-1):
      filtered_new_es.append((l, l+1))
  filtered_new_es.append((len(new_cs)-1, 0))
  new_es = copy.deepcopy(filtered_new_es)
  new_polys[p_ind] = [new_cs, new_es]

  return new_polys

def slide_wall(new_polys, p_ind):

  new_polys = copy.deepcopy(new_polys)
  cs, es = new_polys[p_ind]

  # randomly pick an edge
  k = np.random.choice(range(len(es)))
  e = es[k]
  x0, y0 = cs[e[0]]
  x1, y1 = cs[e[1]]
  dx = abs(x1-x0)
  dy = abs(y1-y0)

  # move
  d = np.random.randint(-8, 8)
  x0_new, y0_new, x1_new, y1_new = int(x0), int(y0), int(x1), int(y1)
  if dx < dy:
      x0_new += d
      x1_new += d
  else:
      y0_new += d
      y1_new += d
  x0_new = np.clip(x0_new, 0, 255)
  y0_new = np.clip(y0_new, 0, 255)
  x1_new = np.clip(x1_new, 0, 255)
  y1_new = np.clip(y1_new, 0, 255)

  # update polygon
  new_polys[p_ind][0][e[0]] = [x0_new, y0_new]
  new_polys[p_ind][0][e[1]] = [x1_new, y1_new]

  return new_polys

def remove_colinear_edges(new_polys, p_ind):
  
  new_polys = copy.deepcopy(new_polys)
  new_cs, new_es = new_polys[p_ind]
  filtered_new_cs = []
  cs_curr = new_cs[0]
  filtered_new_es = []
  for cs_x in new_cs[1:]:
      if np.array_equal(np.array(cs_x), np.array(cs_curr)) == False:
          filtered_new_cs.append(cs_x)
          cs_curr = cs_x
  if np.array_equal(np.array(new_cs[-1]), np.array(new_cs[0])) == False:
      filtered_new_cs.append(new_cs[0])
  for l in range(len(filtered_new_cs)-1):
      filtered_new_es.append((l, l+1))
  filtered_new_es.append((len(filtered_new_cs)-1, 0))
  new_cs = copy.deepcopy(filtered_new_cs)
  new_es = copy.deepcopy(filtered_new_es)
  new_polys[p_ind] = [new_cs, new_es]
  return new_polys

"""def valid_layout(new_polys, p_ind):

  # check for self intersection
  new_cs, new_es = new_polys[p_ind]
  for w in range(len(new_es)):
      for q in range(len(new_es)):
          if q > w:
              ex = new_es[w]
              ey = new_es[q]
              x0, y0 = new_cs[ex[0]]
              x1, y1 = new_cs[ex[1]]
              x2, y2 = new_cs[ey[0]]
              x3, y3 = new_cs[ey[1]]
              if doIntersect(np.array([x0, y0]), np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])): 
                  return False
                  
  # check number of edges
  if len(new_es) < 4:
      return False

  # broken
  for q in range(len(new_es)):
      ex = new_es[q]
      x2, y2 = new_cs[ex[0]]
      x3, y3 = new_cs[ex[1]]
      if (x3 != x2) and (y3 != y2): 
          return False
  return True """
