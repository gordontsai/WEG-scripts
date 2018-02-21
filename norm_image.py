from PIL import Image
import os


os.chdir('..')
cwd=os.getcwd()
im_path=cwd+'/images/'


im=Image.open(im_path+'/Rideshare Heat Map 8.png')
pixels=im.load()


# upper left corner of graph
pixels[117,37]=(255,0,0)
# Lower left corner of graph
pixels[117,733]=(255,0,0)
# upper right corner of graph
pixels[813,37]=(255,0,0)
# Lower right corner of graph
pixels[813,733]=(255,0,0)
im.show()

# box = (100, 100, 400, 400)
# region = im.crop(box)

# region.show()
# rg=region.resize((300,50))

# rg.show()




