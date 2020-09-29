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
import argparse 
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import math
import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import random
from utils import mask_to_bb, ROOM_CLASS, ID_COLOR, draw_graph, draw_masks
sets = {'A':[1, 3], 'B':[4, 6], 'C':[7, 9], 'D':[10, 12], 'E':[13, 100]}

def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Visualization")
    parser.add_argument("--path", required=True,
                        help="dataset path", metavar="DIR")
    #parser.add_argument("--scene", required=True,
    #                    help="scene id", type=int)
    #parser.add_argument("--type", choices=("floorplan", "wireframe", "plane"),
    #                    default="plane", type=str)
    #parser.add_argument("--color", choices=["normal", "manhattan"],
    #                    default="normal", type=str)
    return parser.parse_args()
def filter_graphs(graphs, min_h=0.03, min_w=0.03):
    new_graphs = []
    for g in graphs:
        
        # retrieve data
        rooms_type = g[0]
        rooms_bbs = g[1]
        
        # discard broken samples
        check_none = np.sum([bb is None for bb in rooms_bbs])
        check_node = np.sum([nd == 0 for nd in rooms_type])
        if (len(rooms_type) == 0) or (check_none > 0) or (check_node > 0):
            continue
		
		# hard to edit the graphs now!        
        # # filter small rooms
        # tps_filtered = []
        # bbs_filtered = []
        # for n, bb in zip(rooms_type, rooms_bbs):
        #     h, w = (bb[2]-bb[0]), (bb[3]-bb[1])
        #     if h > min_h and w > min_w:
        #         tps_filtered.append(n)
        #         bbs_filtered.append(bb)
        
        # update graph
        new_graphs.append(g)
    return new_graphs

 
      #def __init__(self, shapes_path, transform=None, target_set=None, split='train'):
#super(Dataset, self).__init__()
#self.shapes_path = shapes_path
#self.split = split:wq

#self.target_set = target_set


def draw_masks_( rms_type, fp_eds, eds_to_rms, im_size=1024):
			import math
			kk=512
			# import webcolors
			# full_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
			rms_masks = []
			fp_mk = np.zeros((kk, kk))
			#print("*****")
			#print(eds_to_rms)
			for k in range(len(rms_type)):
				eds = []
			#	if(rms_type[k]!=4):
			#		continue
				#print("k is",k) 
				for l, e_map in enumerate(eds_to_rms):
					#print(l)
					#print(e_map)
					if k in e_map:
                                               
						eds.append(l)
				rm_im = Image.new('L', (im_size, im_size))
				dr = ImageDraw.Draw(rm_im)
				#print("rm_type",rms_type[k])
				#print("eds is",eds)
				#print(fp_eds[l])
				poly = make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
				poly = [((im_size*x), im_size*y) for x, y in poly]
				if len(poly) >= 2:
					dr.polygon(poly, fill='white')
				rm_arr = np.array(rm_im.resize((kk, kk)))
				inds = np.where(rm_arr>0)
				fp_mk[inds] = k+1

			# trick to remove overlap
			for k in range(len(rms_type)):
				rm_arr = np.zeros((kk, kk))
				inds = np.where(fp_mk==k+1)
				rm_arr[inds] = 1.0
				#print(rms_type[k])
				#print(np.sum(rm_arr))
				rms_masks.append(rm_arr)
				#rg = np.array(rm_im.resize((256,256)))
				#color = ID_COLOR[rms_type[k]]
				#r, g, b = webcolors.name_to_rgb(color)
				#inds = np.array(np.where(rg > 0))
				#reg_deb = np.zeros((256, 256, 3)) 
				#reg_deb[inds[0, :], inds[1, :], :] = [r, g, b] 
				#full_img += reg_deb
			#plt.imsave("imgs/mask_{}_copy_{}_eq_{}.jpg".format(ind,cp,eq),full_img.astype('uint8'))
				## debug
				# rg = np.array(rm_im)
				# color = ID_COLOR[rms_type[k]]
				# r, g, b = webcolors.name_to_rgb(color)
				# inds = np.array(np.where(rg > 0))
			# reg_deb = np.zeros((256, 256, 4))
			# reg_deb[inds[0, :], inds[1, :], :] = [r, g, b, 200]
			# reg_deb = Image.fromarray(reg_deb.astype('uint8')).convert('RGBA')
			# full_img.paste(Image.alpha_composite(full_img, reg_deb))
		# plt.imshow(full_img)
		# plt.show()

			return rms_masks

def make_sequence( edges):
		polys = []
		v_curr = tuple(edges[0][:2])
		e_ind_curr = 0
		e_visited = [0]
		seq_tracker = [v_curr]
		find_next = False
		while len(e_visited) < len(edges):
			if find_next == False:
				if v_curr == tuple(edges[e_ind_curr][2:]):
					v_curr = tuple(edges[e_ind_curr][:2])
				else:
					v_curr = tuple(edges[e_ind_curr][2:])
				find_next = not find_next 
			else:
				# look for next edge
				for k, e in enumerate(edges):
					if k not in e_visited:
						if (v_curr == tuple(e[:2])):
							v_curr = tuple(e[2:])
							e_ind_curr = k
							e_visited.append(k)
							break
						elif (v_curr == tuple(e[2:])):
							v_curr = tuple(e[:2])
							e_ind_curr = k
							e_visited.append(k)
							break

			# extract next sequence
			if v_curr == seq_tracker[-1]:
				polys.append(seq_tracker)
				for k, e in enumerate(edges):
					if k not in e_visited:
						v_curr = tuple(edges[0][:2])
						seq_tracker = [v_curr]
						find_next = False
						e_ind_curr = k
						e_visited.append(k)
						break
			else:
				seq_tracker.append(v_curr)
		polys.append(seq_tracker)

		return polys

def flip_and_rotate(self, v, flip, rot, shape=256.):
		v = self.rotate(np.array((shape, shape)), v, rot)
		if flip:
			x, y = v
			v = (shape/2-abs(shape/2-x), y) if x > shape/2 else (shape/2+abs(shape/2-x), y)
		return v
	
	# rotate coords
def rotate(self, image_shape, xy, angle):
		org_center = (image_shape-1)/2.
		rot_center = (image_shape-1)/2.
		org = xy-org_center
		a = np.deg2rad(angle)
		new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
				-org[0]*np.sin(a) + org[1]*np.cos(a) ])
		new = new+rot_center
		return new

def build_graph( rms_bbs, rms_type, fp_eds, eds_to_rms):

		# create edges
		triples = []
		nodes = rms_type
		bbs = rms_bbs
        
		# encode connections
		for k in range(len(nodes)):
			for l in range(len(nodes)):
				if l > k:
					is_adjacent = any([True for e_map in eds_to_rms if (k in e_map) and (l in e_map)])
					if is_adjacent:
						
							triples.append([k, 1, l])
					
							#triples.append([k, 1, l])
					else:
						
							triples.append([k, -1, l])
						
							#triples.append([k, -1, l])

		# convert to array
		nodes = np.array(nodes)
		triples = np.array(triples)
		bbs = np.array(bbs)
		return bbs, nodes, triples

def _augment(mks):

	flip = random.choice([False, True])
	rot = random.choice([0, 90, 180, 270])
	new_mks = []
	for m in mks:
		m_im = Image.fromarray(m.astype('uint8'))
		m_im = m_im.rotate(rot)
		if flip:
			m_im = m_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
		new_mks.append(np.array(m_im))
	new_mks = np.stack(new_mks)

	return new_mks

def is_adjacent(box_a, box_b, threshold=0.03):
	
	x0, y0, x1, y1 = box_a
	x2, y2, x3, y3 = box_b

	h1, h2 = x1-x0, x3-x2
	w1, w2 = y1-y0, y3-y2

	xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
	yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

	delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
	delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0

	delta = max(delta_x, delta_y)

	return delta < threshold

def one_hot_embedding(labels, num_classes=19):
	"""Embedding labels to one-hot form.

	Args:
	  labels: (LongTensor) class labels, sized [N,].
	  num_classes: (int) number of classes.

	Returns:
	  (tensor) encoded labels, sized [N, #classes].
	"""
	y = torch.eye(num_classes) 
	return y[labels] 
import json
args=parse_args()
line=args.path
#f=open("list.txt","r")
#lines=f.readlines()
#for line in lines:

print("lne",line)
#line=line+"json "
#::wline="40027.json "
with open(line) as f:
		print(line)
		info =json.load(f)
		rms_bbs=np.asarray(info['boxes'])
		fp_eds=info['edges']
		rms_type=info['room_type']
		eds_to_rms=info['ed_rm']
		#print(eds_to_rms)
		#print(rms_type)
		#print(fp_eds)
		#print(rms_bbs)
		s_r=0
		for rmk in range(len(rms_type)):
			if(rms_type[rmk]!=17):
				s_r=s_r+1	
		#print("eds_ro",eds_to_rms)
		rms_bbs = np.array(rms_bbs)/256.0
		fp_eds = np.array(fp_eds)/256.0 
		fp_eds = fp_eds[:, :4]
		tl = np.min(rms_bbs[:, :2], 0)
		br = np.max(rms_bbs[:, 2:], 0)
		shift = (tl+br)/2.0 - 0.5
		rms_bbs[:, :2] -= shift 
		rms_bbs[:, 2:] -= shift
		fp_eds[:, :2] -= shift
		fp_eds[:, 2:] -= shift 
		tl -= shift
		br -= shift
		eds_to_rms_tmp=[]
		for l in range(len(eds_to_rms)):
			eds_to_rms_tmp.append([eds_to_rms[l][0]])
		rooms_bbs, graph_nodes, graph_edges = build_graph(rms_bbs, rms_type, fp_eds, eds_to_rms)
		#graph_img = draw_graph(graph_nodes, graph_edges,0,ind,ed_cp, ed_com)
		rooms_mks = draw_masks_(rms_type, fp_eds, eds_to_rms_tmp,2048)
		graph_nodes = one_hot_embedding(graph_nodes)[:, 1:]
		graph_nodes = torch.FloatTensor(graph_nodes).detach().cpu().numpy()   
		graph_img = draw_graph(graph_nodes, graph_edges,0,str(s_r)+"_"+line) 
		#print(len(rooms_mks))
		draw_masks(rooms_mks, graph_nodes,2048,str(s_r)+"_"+line)
		with open("json_data_update/"+line[:-5]+".json","w") as fkn:
			 json.dump(info, fkn) 
