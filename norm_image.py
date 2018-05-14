from PIL import Image
import os
from dimensions import dim


os.chdir('..')
cwd=os.getcwd()
im_path=cwd+'/images/'


im=Image.open(im_path+'/Rideshare Heat Map 8.png')
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
for y in range(10):
  box_list=[]
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
    pixels[left,up]=(0,0,0)
    pixels[left,down]=(0,0,0)
    pixels[right,up]=(0,0,0)
    pixels[right,down]=(0,0,0)

    #assign to list in order
    box_list.append(region)
    #iterate
    box_num+=1

im.show()

#display boxes
# for img in box_list:
  # img.show()


print(sum(dim[0]))
# box = (100, 100, 400, 400)
# region = im.crop(box)

# region.show()
# rg=region.resize((300,50))

# rg.show()




