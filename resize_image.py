from PIL import Image  
import os
# Opens a image in RGB mode  

for i in range(1,184):
	image_name = "image"+str(i)+".png"
	path = os.path.join("data","custom","images",image_name)
	im = Image.open(path)  
	newsize = (400,400)
	im1 = im.resize(newsize) 
	im1.save(path)
