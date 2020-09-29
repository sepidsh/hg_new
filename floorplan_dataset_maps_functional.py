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
from fp1 import reader
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
from utils import mask_to_bb, ROOM_CLASS, ID_COLOR
sets = {'A':[1, 3], 'B':[4, 6], 'C':[7, 9], 'D':[10, 12], 'E':[13, 100]}
log_file = open("./log.txt","w")
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
		
        # update graph
        new_graphs.append(g)
    return new_graphs

class FloorplanGraphDataset(Dataset):
	def __init__(self, shapes_path, transform=None, target_set=None, split='train'):
		super(Dataset, self).__init__()
		#self.shapes_path = shapes_path
		self.split = split
		self.subgraphs=[]
		#self.target_set = target_set
		f1=open("list2.txt","r")
		f=open("output3.txt", "a+") 
		lines=f1.readlines()
		h=0
		for line in lines:
			a=[]
			h=h
			line="json_data_update/"+line
			if(split=='train'):
				if(h%1==0):
					with open(line[:-1]) as f2:
						rms_type, fp_eds,rms_bbs,eds_to_rms,eds_to_rms_tmp=reader(line[:-1]) 
						if(len([x for x in rms_type if x != 15 and x != 17]) == 8):
							continue
						a.append(rms_type)
						a.append(rms_bbs)
						a.append(fp_eds)
						a.append(eds_to_rms)
						a.append(eds_to_rms_tmp)
						f.write(str(line))
						f.write("  ")
						f.write(str(len(rms_type)))
						f.write("   ")
						f.write(str(rms_type).strip('[]')) 
						f.write("   ********   ")
						f.write(str(len(eds_to_rms)))
						f.write("\n")

			
					self.subgraphs.append(a)

			if(split=='eval'):
				if(h%1==0) :
					with open(line[:-1]) as f2:
						
						rms_type, fp_eds,rms_bbs,eds_to_rms,eds_to_rms_tmp=reader(line[:-1]) 
						if(len([x for x in rms_type if x != 15 and x != 17]) != 8):
							continue
						a.append(rms_type)
						a.append(rms_bbs)
						a.append(fp_eds)
						a.append(eds_to_rms)
						a.append(eds_to_rms_tmp)
						f.write(str(line))
						f.write("  ")
						f.write(str(len(rms_type)))
						f.write("   ")
						f.write(str(rms_type).strip('[]')) 
						f.write("  ********  ")
						f.write(str(len(eds_to_rms)))
						f.write("\n")
					self.subgraphs.append(a)
			
		if split == 'train':
			self.augment = True
		elif split == 'eval':
			self.augment = False
		else:
			print('Error split not supported')        
			exit(1)
		self.transform = transform
		f.close() 
		print(len(self.subgraphs))   

	def __len__(self):
		return len(self.subgraphs)

	def __getitem__(self, index):

		graph = self.subgraphs[index]
		rms_type = graph[0]
		rms_bbs = graph[1]
		fp_eds = graph[2]
		eds_to_rms= graph[3]
		eds_to_rms_tmp=graph[4]
		rms_bbs = np.array(rms_bbs)
		fp_eds = np.array(fp_eds)

		# extract boundary box and centralize
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

		# build input graph
		graph_nodes, graph_edges, rooms_mks = self.build_graph(rms_type, fp_eds, eds_to_rms)

		# convert to tensor
		graph_nodes = one_hot_embedding(graph_nodes)[:, 1:]
		graph_nodes = torch.FloatTensor(graph_nodes)
		graph_edges = torch.LongTensor(graph_edges)
		rooms_mks = torch.FloatTensor(rooms_mks)
		rooms_mks = self.transform(rooms_mks)
		return rooms_mks, graph_nodes, graph_edges


	def draw_masks(self, rms_type, fp_eds, eds_to_rms, im_size=256):
		import math
		# import webcolors
		# full_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
		rms_masks = []
		fp_mk = np.zeros((32, 32))
		for k in range(len(rms_type)):
			eds = []
			for l, e_map in enumerate(eds_to_rms):
				if k in e_map:
					eds.append(l)
			rm_im = Image.new('L', (im_size, im_size))
			rm_im = rm_im.filter(ImageFilter.MaxFilter(7))
			dr = ImageDraw.Draw(rm_im)
			poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
			poly = [(im_size*x, im_size*y) for x, y in poly]
			if len(poly) >= 2:
				dr.polygon(poly, fill='white')
			rm_im = rm_im.resize((32, 32)).filter(ImageFilter.MaxFilter(3))
			rm_arr = np.array(rm_im)
			inds = np.where(rm_arr>0)
			fp_mk[inds] = k+1

		# trick to remove overlap
		for k in range(len(rms_type)):
			rm_arr=np.ones((32,32))
			rm_arr = np.zeros((32, 32))
			inds = np.where(fp_mk==k+1)
			rm_arr[inds] = 1.0
			rms_masks.append(rm_arr)

		plt.figure()
		debug_arr = np.sum(np.array(rms_masks), 0)
		debug_arr[debug_arr>0] = 255
		im = Image.fromarray(debug_arr)
		plt.imshow(im)
		plt.show()

		return rms_masks

	def make_sequence(self, edges):
		polys = []
		#print(edges)
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

	def build_graph(self, rms_type, fp_eds, eds_to_rms):

		# create edges
		triples = []
		nodes = [x for x in rms_type if x != 15 and x != 17]
		doors=[x for x in rms_type if x==15 or x==17]
		# doors to rooms
		doors_inds = [
		]
		for k, r in enumerate(rms_type):
			if r in [15, 17]:
				doors_inds.append(k)

		# for each door compare against all rooms
		door_to_rooms = defaultdict(list)
		for d in doors_inds:
			door_edges = eds_to_rms[d]
			for r in range(len(nodes)):
				if r not in doors_inds:
					is_adjacent = any([True for e_map in eds_to_rms if (r in e_map) and (d in e_map)])
					if is_adjacent:
						door_to_rooms[d].append(r)


		# encode connections
		for k in range(len(rms_type)):
			for l in range(len(rms_type)):
				if l > k:
					is_adjacent = any([True for d_key in door_to_rooms if (k in door_to_rooms[d_key]) and (l in door_to_rooms[d_key])])
					if is_adjacent:
						if 'train' in self.split:
							triples.append([k, 1, l])
						else:
							triples.append([k, 1, l])
					else:
						if 'train' in self.split:
							triples.append([k, -1, l])
						else:
							triples.append([k, -1, l])

		# get rooms masks
		eds_to_rms_tmp = []
		for l in range(len(eds_to_rms)):                  
			eds_to_rms_tmp.append([eds_to_rms[l][0]])

		rms_masks = []
		im_size = 256
		kk=128
		fp_mk = np.zeros((kk, kk))
		print("nodes",nodes)
		for k in range(len(nodes)):
			eds = []
			for l, e_map in enumerate(eds_to_rms_tmp):
				if (k in e_map) and (15 not in e_map) and (17 not in e_map):
					eds.append(l)

			rm_im = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(rm_im)
			for l in eds:
				x0, y0, x1, y1 = fp_eds[l] * 256
				dr.line((x0, y0, x1, y1), 'white', width=1)

			poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
			poly = [(im_size*x, im_size*y) for x, y in poly]
			if len(poly) >= 2:
				dr.polygon(poly, fill='white')
			else:
				print("Empty room")
				exit(0)


			rm_im = rm_im.resize((kk, kk))#.filter(ImageFilter.MaxFilter(3))
			rm_arr = np.array(rm_im)
			inds = np.where(rm_arr>0)
			fp_mk[inds] = k+1

		# trick to remove overlap
		for k in range(len(nodes)):
			rm_arr = np.zeros((kk, kk))
			inds = np.where(fp_mk==k+1)
			rm_arr[inds] = 1.0
			rms_masks.append(rm_arr)

		for h in range(len(doors)):
			eds = []
			k=len(nodes)+h
			for l, e_map in enumerate(eds_to_rms_tmp):
				if (k in e_map):# and ((15  in e_map) or (17  in e_map)):
					eds.append(l)

			rm_im_2 = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(rm_im_2)
			for l in eds:
				x0, y0, x1, y1 = fp_eds[l] * 256
				dr.line((x0, y0, x1, y1), 'white', width=1)

			poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
			poly = [(im_size*x, im_size*y) for x, y in poly]
			if len(poly) >= 2:
				dr.polygon(poly, fill='white')
			else:
				print("Empty room")
				exit(0)


			rm_im_2 = rm_im_2.resize((kk, kk))#.filter(ImageFilter.MaxFilter(3))
			rm_arr = np.array(rm_im_2)
			inds = np.where(rm_arr>0)
			fp_mk[inds] = k+1

		# trick to remove overlap
		#for k in range(len(nodes)):
		#	rm_arr = np.zeros((kk, kk))
		#	inds = np.where(fp_mk==k+1)
		#	rm_arr[inds] = 1.0
			rms_masks.append(rm_arr)



		# convert to array
		nodes = np.array(nodes)
		triples = np.array(triples)
		rms_masks = np.array(rms_masks)

		return nodes, triples, rms_masks

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
	#print(" label is",labels) 
	return y[labels] 

def floorplan_collate_fn(batch):
	all_rooms_mks, all_nodes, all_edges = [], [], []
	all_node_to_sample, all_edge_to_sample = [], []
	node_offset = 0
	eds_sets=[]
	for i, (rooms_mks, nodes, edges) in enumerate(batch):
		O, T = nodes.size(0), edges.size(0)
		all_rooms_mks.append(rooms_mks)
		all_nodes.append(nodes)
		#eds_sets.append(eds_set)
		edges = edges.clone()
		if edges.shape[0] > 0:
			edges[:, 0] += node_offset
			edges[:, 2] += node_offset
			all_edges.append(edges)
		all_node_to_sample.append(torch.LongTensor(O).fill_(i))
		all_edge_to_sample.append(torch.LongTensor(T).fill_(i))
		node_offset += O
	all_rooms_mks = torch.cat(all_rooms_mks, 0)
	all_nodes = torch.cat(all_nodes)
	if len(all_edges) > 0:
		all_edges = torch.cat(all_edges)
	else:
		all_edges = torch.tensor([])       
	all_node_to_sample = torch.cat(all_node_to_sample)
	all_edge_to_sample = torch.cat(all_edge_to_sample)
	return all_rooms_mks, all_nodes, all_edges, all_node_to_sample, all_edge_to_sample

