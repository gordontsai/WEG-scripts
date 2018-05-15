from PIL import Image
import os
from dimensions import dim

def get_cum_size(box_size_list,gbox):
  box_cum_left_list=[]
  box_cum_up_list=[]
  initial_up_cum=0
  for y in range(10):
    initial_up_cum+=box_size_list[y][0][1]
    # print(box_size_list[y][0])
    box_cum_up_list.append(initial_up_cum)
    # print(box_cum_up_list)
  initial_left_cum=0
  for x in range(10):
    initial_left_cum+=box_size_list[0][x][0]
    # print(box_size_list[0][x])
    box_cum_left_list.append(initial_left_cum)
    # print(box_cum_left_list)

  return box_cum_left_list, box_cum_up_list

def get_boxes_size(boxes):
  box_list=[]
  for row in enumerate(boxes):
    box_row=[]

    for box in enumerate(row[1]):
      box_row.append(box[1].size)
    box_list.append(box_row)
  return box_list

#gbox stands for graph box
def show_resized(im,resized,gbox):

  #Change all to black for the big graph box
  pixels=im.load()
  for y in range(gbox[1],gbox[3]+1):
    for x in range(gbox[0],gbox[2]+1):
      pixels[x,y]=(0,0,0)

  box_size_list=get_boxes_size(resized)
  #These should match
  # print(box_size_list[1][0])
  # print(resized[1][0].size)
  left_cum, up_cum=get_cum_size(box_size_list,gbox)

  box_coord_list=[]
  for row in enumerate(box_size_list):
    box_coord_row=[]
    if row[0]==0:
      up=gbox[1]
    else:
      up=gbox[1]+up_cum[row[0]-1]
    for box in enumerate(row[1]):
      if box[0]==0:
        left=gbox[0]
      else:
        left=gbox[0]+left_cum[box[0]-1]
      right=left+left_cum[box[0]]
      down=up+up_cum[row[0]]
      # coord=(left,up,right,down)
      coord=(left,up)
      box_coord_row.append(coord)
    box_coord_list.append(box_coord_row)

  pixels=im.load()
  for y in range(10):
    for x in range(10):
      # resized[y][x].show()

      # coord=box_coord_list[y][x]
      if x==0 and y==0:
        coord_short=(gbox[0],gbox[1])
      elif x==0:
        coord_short=(gbox[0],gbox[1]+up_cum[y-1])
      elif y ==0:
        coord_short=(gbox[0]+left_cum[x-1],gbox[1])
      else:
        coord_short=(gbox[0]+left_cum[x-1],gbox[1]+up_cum[y-1])

      # coord_short=(coord[0],coord[1])
      im.paste(resized[y][x],coord_short)
      # pixels[coord_short[0],coord_short[1]]=(255,255,255)

  for x in range(gbox[0],gbox[2]+1):
    pixels[x,gbox[3]-2]=(255,255,255)
    pixels[x,gbox[3]-1]=(255,255,255)
    pixels[x,gbox[3]]=(255,255,255)

  im.show()




#Boxes is a list of the 100 images cut
#dim is read in with distributions for the x and y axis
## dim is car price distribution then time_worth distribution, which is dim_x
##then dim_y
def resize_boxes(im,boxes,d,gbox):
  boxes_resized=[]
  graph=im.crop(gbox)
  for row in enumerate(boxes):
    boxes_resized_row=[]
    for box in enumerate(row[1]):
      x_new=int(round(graph.size[0]*dim[0][box[0]],0))
      y_new=int(round(graph.size[1]*dim[1][row[0]],0))
      # if x_new ==0:
        # x_new=1
      # if y_new == 0:
        # y_new=1
      boxes_resized_row.append(box[1].resize((x_new,y_new)))
    boxes_resized.append(boxes_resized_row)
  return boxes_resized


#Pass in object from im.load() and left, right, up down coordinates
#this marks the four corners of the box as black
def mark_image(pixels,left,right,up,down):
  pixels[left,up]=(0,0,0)
  pixels[left,down]=(0,0,0)
  pixels[right,up]=(0,0,0)
  pixels[right,down]=(0,0,0)




os.chdir('..')
cwd=os.getcwd()
im_path=cwd+'/images/'


im=Image.open(im_path+'/Rideshare Heat Map 9.png')
pixels=im.load()


# Finding the pixel boundaries of the image
# upper left corner of graph
# pixels[117,37]=(255,0,0)
# # Lower left corner of graph
# pixels[117,733]=(255,0,0)
# # upper right corner of graph
# pixels[813,37]=(255,0,0)
# # Lower right corner of graph
# pixels[813,733]=(255,0,0)


# Setting Points
lower_bnd=733
upper_bnd=37
left_bnd=117
right_bnd=813
len_left_right=right_bnd-left_bnd
len_lower_upper=lower_bnd-upper_bnd
p_left_lower=(left_bnd,lower_bnd)
p_right_lower=(right_bnd,lower_bnd)
p_left_upper=(left_bnd,upper_bnd)
p_right_upper=(right_bnd,upper_bnd)
graph_box=(left_bnd,upper_bnd,right_bnd,lower_bnd)


# Cut into 10 x 10
# input for crop is a tuple defining (left, upper, right, lower)
num_boxes=10
xstep=int(round(len_left_right/num_boxes,0))
ystep=int(round(len_lower_upper/num_boxes,0))
xextra=xstep-(len_left_right%xstep)
yextra=ystep-(len_lower_upper%ystep)
orig_box=(left_bnd,upper_bnd,right_bnd,lower_bnd)
# Bottom left most 10 x 10 box
# box=(left_bnd,lower_bnd-ystep*1,left_bnd+xstep*1,lower_bnd)
# Upper left most 10 x 10 box

#store the ten boxes
box_num=1
box_list=[]
for y in range(10):
  box_row=[]
  for x in range(10):
    if x==9 and y==9:
      left=left_bnd+xstep*(x)
      up=upper_bnd+ystep*(y)
      right=left_bnd+xstep*(x+1)-xextra
      down=upper_bnd+ystep*(y+1)-yextra
    elif x ==9:
      left=left_bnd+xstep*(x)
      up=upper_bnd+ystep*(y)
      right=left_bnd+xstep*(x+1)-xextra
      down=upper_bnd+ystep*(y+1)
    elif y==9:
      left=left_bnd+xstep*(x)
      up=upper_bnd+ystep*(y)
      right=left_bnd+xstep*(x+1)
      down=upper_bnd+ystep*(y+1)-yextra
    else:
      left=left_bnd+xstep*(x)
      up=upper_bnd+ystep*(y)
      right=left_bnd+xstep*(x+1)
      down=upper_bnd+ystep*(y+1)
    box=(left,up,right,down)
    # print('box ',str(box_num),box)
    region=im.crop(box)
    #Use this function to check if it has marked and saved the right boxes
    # mark_image(pixels,left,right,up,down)
    #assign row to a list
    box_row.append(region)
    #iterate
    box_num+=1
  box_list.append(box_row)


resized=resize_boxes(im,box_list,dim,graph_box)

show_resized(im,resized,graph_box)
