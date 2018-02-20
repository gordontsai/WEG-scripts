from PIL import Image
import os


os.chdir('..')
cwd=os.getcwd()
im_path=cwd+'/images/'


im=Image.open(im_path+'/Rideshare Heat Map 5.png')


box = (100, 100, 400, 400)
region = im.crop(box)

region.show()
rg=region.resize((300,50))

rg.show()




