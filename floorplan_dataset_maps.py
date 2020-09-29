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
#item_no=0
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

class FloorplanGraphDataset(Dataset):
	def __init__(self, shapes_path, transform=None, target_set=None, split='train'):
		super(Dataset, self).__init__()
		#self.shapes_path = shapes_path
		self.split = split
		self.subgraphs=[]
		#elf.target_set = target_set
		hp=open("list2.txt","r")
		#f=open("output3.txt", "a+") 
		lines=hp.readlines()
		h=0
		for line in lines:
			a=[]
			h=h+1
			#print(line)
			line="json_data_update/"+line	
			if(split=='train'):
				if(h>=-1):
					with open(line[:-1]) as f2:
						rms_type, fp_eds,rms_bbs,eds_to_rms,eds_to_rms_tmp=reader(line[:-1]) 
						#info=json.load(f2)
						
						rms_no=-1
						for kn in range(len(rms_type)):
							if(rms_type[kn]!=17):
								rms_no=rms_no+1

						if(rms_no==7):
							continue
						#rms_bbs=(info['boxes'])
						#fp_eds=info['edges']
						#rms_type=info['room_type']
						#rm_ip=[]
						#for r_id in range(len(rms_type)):
						#	if(rms_type[r_id]>=17):
						#		rm_ip.append(r_id)
						#if(len(rms_type)>=24):
						#	continue
						#eds_to_rms=info['ed_rm']
						#print("Eds_type",rms_type)
						#print("fp_eds", pfp_eds)
						#print("eds_to_rms",eds_to_rms)
						


						#eds_to_rms_nodoor=[]
						"""for ed_id in range(len(eds_to_rms)):
							fn=10000
							for i_d in range(len(rm_ip)):
							
								if(eds_to_rms[ed_id][0]==rm_ip[i_d])
									ed_tmp=[]
									if(len(eds_to_rms[ed_id])>1)
										tp=eds_to_rms[1]

								
							if(fn!=0):
								eds_to_rms_nodoor.append(eds_to_rms[ed_id])
										

						#eds_to_rms=info['ed_rm']"""
						a.append(rms_type)
						a.append(rms_bbs)
						a.append(fp_eds)
						a.append(eds_to_rms)
						a.append(eds_to_rms_tmp)
						#f.write(str(line))
						
						#f.write("  ")
						#f.write(str(len(rms_type)))
						#f.write("   ")
						

						#f.write(str(rms_type).strip('[]')) 
						#f.write("   ********   ")

						
						#f.write(str(len(eds_to_rms)))
						#f.write("\n")

			
					self.subgraphs.append(a)

			if(split=='eval'):
				if(h<=20000) :
					with open(line[:-1]) as f2:
						
						#rms_bbs=(info['boxes'])
						rms_type, fp_eds,rms_bbs,eds_to_rms,eds_to_rms_tmp=reader(line[:-1]) 
						#info=json.load(f2)
						rms_no=-1
						for kn in range(len(rms_type)):
							if(rms_type[kn]!=17):
								rms_no=rms_no+1
						if(rms_no!=7):
							continue
						#rms_bbs=(info['boxes'])
						#fp_eds=info['edges']
						#rms_type=info['room_type']
						#rm_ip=[]
						#for r_id in range(len(rms_type)):
						#	if(rms_type[r_id]>=17):
						#		rm_ip.append(r_id)
						#if(len(rms_type)>=24):
						#	continue
						#eds_to_rms=info['ed_rm']
						#print("Eds_type",rms_type)
						#print("fp_eds", fp_eds)
						#print("eds_to_rms",eds_to_rms)
						


						#eds_to_rms_nodoor=[]
						"""for ed_id in range(len(eds_to_rms)):
							fn=10000
							for i_d in range(len(rm_ip)):
							
								if(eds_to_rms[ed_id][0]==rm_ip[i_d])
									ed_tmp=[]
									if(len(eds_to_rms[ed_id])>1)
										tp=eds_to_rms[1]

								
							if(fn!=0):
								eds_to_rms_nodoor.append(eds_to_rms[ed_id])
										

						#eds_to_rms=info['ed_rm']"""
						a.append(rms_type)
						a.append(rms_bbs)
						a.append(fp_eds)
						a.append(eds_to_rms)
						a.append(eds_to_rms_tmp)
						#f.write(str(line))
						#f.write("  ")
						#f.write(str(len(rms_type)))
						#f.write("   ")
						#f.write(str(rms_type).strip('[]')) 
						#f.write("  ********  ")
						#f.write(str(len(eds_to_rms)))
						#f.write("\n")
					self.subgraphs.append(a)
			
		#self.subgraphs = np.load('data_v2.npy'.format(self.shapes_path), allow_pickle=True)
		if split == 'train':
			self.augment = True
		elif split == 'eval':
			self.augment = False
		else:
			print('Error split not supported')        
			exit(1)
		self.transform = transform
		#self.subgraphs = filter_graphs(self.subgraphs)
		#f.close() 
		# filter samples
		"""min_N = sets[self.target_set][0]
		max_N = sets[self.target_set][1]
		filtered_subgraphs = []
		for g in self.subgraphs:
			rooms_type = g[0]    
			in_set = (len(rooms_type) >= min_N) and (len(rooms_type) <= max_N)
			if (split == 'train') and (in_set == False):
				filtered_subgraphs.append(g)
			elif (split == 'eval') and (in_set == True):
				filtered_subgraphs.append(g)
		self.subgraphs = filtered_subgraphs
		if split == 'eval':
			self.subgraphs = self.subgraphs[:5000] # max 5k"""
		print(len(self.subgraphs))   
		item_no=0 
		"""# doblecheck
		deb_dic = defaultdict(int)
		for g in self.subgraphs:
			rooms_type = g[0] 
			if len(rooms_type) > 0:
				deb_dic[len(rooms_type)] += 1
		print("target samples:", deb_dic)"""
        
	def __len__(self):
		return len(self.subgraphs)

	def __getitem__(self, index):
		#item_no=item_no+1
		#print(index)
		# load data # r_types, bbs, edges, ed_to_rs, dr_to_ed
		graph = self.subgraphs[index]
		rms_type = graph[0]
		#rms_type_new=[]
		#for i in range(len(rms_type)):
		#	if(rms_type[i]!=17)& (rms_type[i]!=15):
		#		rms_type_new.append(rms_type[i])
		#rms_type.append(0)
		rms_bbs = graph[1]
		#rms_bbs.append(np.array([0,0,255,255]))
		rms_type_new=[]
		rms_bbs_new=[]
		rms_type_new=[]

		for i in range(len(rms_type)):
			if(rms_type[i]!=17)& (rms_type[i]!=15):
				rms_type_new.append(rms_type[i])
				rms_bbs_new.append(rms_bbs[i])
			#else:
			#	rms_type_new.append(18)
		if(len(rms_type_new)==0):
			print("len",len(rms_type_new))
		
		
		fp_eds = graph[2]
		eds_to_rms= graph[3]
		eds_to_rms_tmp=graph[4]
		#drs_to_eds=graph[4]
		rms_no=-1
		eds_to_rms_new=[]
		for kn in range(len(rms_type)):
			if(rms_type[kn]!=17):
				rms_no=rms_no+1
		for i in range (len(eds_to_rms)):
			if(eds_to_rms[i][0]>=rms_no):
				eds_to_rms_new.append(eds_to_rms[i])
		#eds_to_rms=eds_to_rms_tmp
		#drs_to_eds = graph[4]
		#print(rms_type,"rms_type")
		#print(rms_bbs,"rms_bbs")
		#print(fp_eds,"fp_eds")
		#print(eds_to_rms,"ed_rms")
		

		"""eds= []
		for i in range(len(rms_type)):
			for j in range(len(rms_type)):
				if(j<=i):
					continue
				eds.append([i,j])

		"""
		#print(drs_to_eds,"drs_to_eds")
		#sort_tmps=drs_to_eds
		#sort_tmps.sort()
		#print(sort_tmps,"sort")
		# has to augment edges too!
		# if self.augment:
		# 	rot = random.randint(0, 3)*90.0
		# 	flip = random.randint(0, 1) == 1
		# 	rooms_bbs_aug = []
		# 	for bb in rooms_bbs:
		# 		x0, y0 = self.flip_and_rotate(np.array([bb[0], bb[1]]), flip, rot)
		# 		x1, y1 = self.flip_and_rotate(np.array([bb[2], bb[3]]), flip, rot)
		# 		xmin, ymin = min(x0, x1), min(y0, y1)
		# 		xmax, ymax = max(x0, x1), max(y0, y1)
		# 		rooms_bbs_aug.append(np.array([xmin, ymin, xmax, ymax]).astype('int'))
		# 	rooms_bbs = rooms_bbs_aug
		# rooms_bbs = np.stack(rooms_bbs)

		# normalize [0, 1]
		"""k_p=fp_eds  
		rms_bbs = np.array(rms_bbs)/256.0
		fp_eds = np.array(fp_eds)/256.0
		#k_p=fp_eds
		fp_eds = fp_eds[:, :4]

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
		# boundary_bb = np.concatenate([tl, br])

		# build input graph"""
		rooms_bbs, graph_nodes, graph_edges,rooms_mks = self.build_graph(rms_bbs, rms_type, fp_eds, eds_to_rms)
		#rooms_mks = self.draw_masks_2(rms_type, fp_eds, eds_to_rms_tmp,rms_no)
		#print("lenenen",len(graph_nodes))
		# convert to tensor
		#print("graph nodes is", graph_nodes)
		#print("one hot",one_hot_embedding(graph_nodes))
		graph_nodes = one_hot_embedding(graph_nodes)[:, 1:]
		#print("graph nodes is !!!!", graph_nodes)
		graph_nodes = torch.FloatTensor(graph_nodes)
		graph_edges = torch.LongTensor(graph_edges)
		rooms_mks = torch.FloatTensor(rooms_mks)
		rooms_mks = self.transform(rooms_mks)
		#value, ind=torch.max(eds_set,0)
		#log_file.write("{}\n".format(ind))
		#log_file.flush()
		#print("eds_set 1!",eds_set)
		#for i in range(10):	
		#for b in range(len(eds_set)):
		#	if(eds_set[b]>=12):
		#		print(eds_set[b])
		#		print(eds_set)
		#		print(k_p)
		#eds_set=one_hot_embedding(eds_set,20)
		#print("eds_set",eds_set)
		return rooms_mks, graph_nodes, graph_edges


	def draw_masks_2(self, rms_type, fp_eds, eds_to_rms,rm_no, im_size=64):
		import math
		# import webcolors
		# full_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
		rms_masks = []
		fp_mk = np.zeros((64, 64))
		for k in range(len(rms_type)):
			#if(rms_type[k]==0):
			#	rm_arr=np.ones((32,32))
			#	continue
			eds = []
			for l, e_map in enumerate(eds_to_rms):
				if k in e_map:
					eds.append(l)
			rm_im = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(rm_im)
			poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
			poly = [((im_size*x), im_size*y) for x, y in poly]
			#print(rms_type[k])
			#print(poly)
			#print("****")
			#if(len(poly)==2):
			#	if(poly[1][0]==poly[0][0]):
			#		poly.append( (poly[0][0]-1, poly[0][1]))  
			#		poly.append( (poly[1][0]-1, poly[1][1]))
			#	if(poly[1][1]==poly[0][1]):  
			#		poly.append( (poly[0][0], poly[0][1]-1))  
			#		poly.append( (poly[1][0], poly[1][1]-1))     
			#	#print(poly)
			##	#print(poly[1][0],")))")
			if len(poly) >= 2:
				dr.polygon(poly, fill='white')
			rm_arr = np.array(rm_im.resize((64, 64)))
			inds = np.where(rm_arr>0)
			fp_mk[inds] = k+1
					


		# trick to remove overlap
		#for k in range(len(rms_type)):
		#	rm_arr=np.zeros((128,128))
		#	inds = np.where(fp_mk==k+1)
		#	rm_arr[inds] = 1.0
			rms_masks.append(rm_arr)

		return rms_masks

	def draw_masks(self, rms_type, fp_eds, eds_to_rms,room_no, im_size=256):
		
			import math
			kk=128
			im_size=512
			
			#for i in range(len(rms_type)):

			# import webcolors
			# full_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
			rms_masks = []
			fp_mk = np.zeros((kk, kk))
			fp_value=np.zeros((kk,kk))
			#print("*****")
			#print(eds_to_rms)
			
			for k in range(room_no):
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
				poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
				poly = [((im_size*x), (im_size*y)) for x, y in poly]
			
				#print("poly",poly)
				if len(poly) >= 2:
					dr.polygon(poly,fill="white")
				rm_arr = np.array(rm_im.resize((kk,kk)))
				inds=np.where(rm_arr>5)	
				#fp_mk[inds]=k+1
				#print(inds)
				#print(len(inds[0]))
				for i in range(len(inds[0])):
					for j in range(len(inds[1])):
						if(fp_value[inds[0][i]][inds[1][j]]<rm_arr[inds[0][i]][inds[1][j]]):
							fp_value[inds[0][i]][inds[1][j]]=rm_arr[inds[0][i]][inds[1][j]]
							fp_mk[inds[0][i]][inds[1][j]]=k+1	



			# trick to remove overlap
			for k in range(room_no):
				rm_arr = np.zeros((kk, kk))
				inds = np.where(fp_mk==k+1)
				rm_arr[inds] = 1.0
				rm_arr[inds] = 1.0
				#print(rms_type[k])
				#print(np.sum(rm_arr))
				rms_masks.append(rm_arr)
				#rg = np.array(rm_im.resize((256,256)))
				##plt.imshow(rm_arr_new_2)
				#plt.imshow(rm_arr_3)
			for k in range(room_no,len(rms_type)):
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
				#print(fp_eds[l]
				poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
				poly = [((im_size*x), (im_size*y)) for x, y in poly]
			
			#	print("dooor",poly)
				if len(poly) >= 2:
					dr.polygon(poly,fill="white")
				rm_arr = np.array(rm_im.resize((kk,kk)))
				
				rm_arr_tmp=rm_arr.copy()
				#rm_arr=rm_arr_tmp.copy()
				
				for i in range(1,kk-1):
					for j in range(1,kk-1):
						if(rm_arr[i][j]==0):
							cor=0
							if((rm_arr[i][j+1])>0) &( (rm_arr[i-1][j+1])>0) &((rm_arr[i-1][j])>0):
								cor=cor+1
							if((rm_arr[i+1][j])>0) & ((rm_arr[i+1][j-1])>0) &((rm_arr[i][j-1])>0):
								cor=cor+1
							if((rm_arr[i-1][j-1])>0) &( (rm_arr[i][j-1])>0) &((rm_arr[i-1][j])>0):
								cor=cor+1
							if((rm_arr[i][j+1])>0) & ((rm_arr[i+1][j+1])>0) &((rm_arr[i+1][j])>0):
								cor=cor+1
							#if(rm_arr[i+1][j])>0):
							#	cor=cor+1
							if(cor==1):
								rm_arr_tmp[i][j]=255
						

						
				inds=np.where(rm_arr<0)
				#print(rms_type[k])
				#print(np.sum(rm_arr))
				rms_masks.append(rm_arr_tmp)
				#rg = np.array(rm_im.resize((256,256)))
				##plt.imshow(rm_arr_new_2)
				#plt.imshow(rm_arr_3)
		# plt.show()

				inds=np.where(rm_arr<0)
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

	def build_graph(self, rms_bbs, rms_type, fp_eds, eds_to_rms):

		#nodes= [x for x in rms_type if x != 15 and x != 17]
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
						if 'train' in self.split:
							triples.append([k, 1, l])
						else:
							triples.append([k, 1, l])
					else:
						if 'train' in self.split:
							triples.append([k, -1, l])
						else:
							triples.append([k, -1, l])

		# convert to array
		nodes = np.array(nodes)
		#print(triples,"tri")
		triples = np.array(triples)
		bbs = np.array(bbs)
		#print(nodes)

		eds_to_rms_tmp = []
		for l in range(len(eds_to_rms)):                  
			eds_to_rms_tmp.append([eds_to_rms[l][0]])

		rms_masks = []
		im_size = 256
		kk=64
		fp_mk = np.zeros((kk, kk))
		#print("nodes",nodes)
		nodes_2= [x for x in rms_type if x != 15 and x != 17]
		#doors=[x for x in rms_type if x==15 or x==17]
		for k in range(len(nodes)):
			eds = []
			for l, e_map in enumerate(eds_to_rms_tmp):
				if (k in e_map):# and (15 not in e_map) and (17 not in e_map):
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
			##rms_masks.append(rm_arr)
			plt.imshow(rm_arr)
                        #plt.show()
			plt.show()
			rm_arr[inds] = 1
			
			plt.imshow(rm_arr)
                        #plt.show()
			plt.show()
			rms_masks.append(rm_arr)
		# trick to remove overlap
		"""for k in range(len(nodes_2)):
			rm_arr = np.zeros((kk, kk))
			inds = np.where(fp_mk==k+1)
			rm_arr[inds] = 1.0"""
			#rms_masks.append(rm_arr)

		"""for h in range(len(doors)):
			eds = []
			k=len(nodes_2)+h
			for l, e_map in enumerate(eds_to_rms_tmp):
				if (k in e_map):# and ((15  in e_map) or (17  in e_map)):
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
		for k in range(len(doors)):
			rm_arr = np.zeros((kk, kk))
			inds = np.where(fp_mk==k+1)
			rm_arr[inds] = 1.0
			rms_masks.append(rm_arr)"""
		return bbs, nodes, triples,rms_masks

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

