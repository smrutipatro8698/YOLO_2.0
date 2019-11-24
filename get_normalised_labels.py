from PIL import Image
import os

for i in range(1,184):
	image_name = "image"+str(i)+".png"
	label_name = "labels"+str(i)+".txt"

	new_label_name = "image"+str(i)+".txt"
	#print(image_name)
	path = os.path.join("train_images","clean",image_name)
	path1 = os.path.join("train_labels","clean",label_name)
	im = Image.open(path)
	width, height = im.size
	f = open(path1, "r")

	new_path = os.path.join("train_labels",new_label_name)
	fp = open(new_path,"w")
	for x in f:
		x1 = x.split(" ")
		x1 = x1[0:5]
		for a in range(0,len(x1)):
			if (a==0):
				fp.write(str(x1[a]))
			if (a==1):
				fp.write(str(float(x1[a])/float(width)))
			if (a==2):
				fp.write(str(float(x1[a])/float(height)))
			if (a==3):
				fp.write(str(float(x1[a])/float(width)))
			if (a==4):
				fp.write(str(float(x1[a])/float(height)))
			if(a!=4):
				fp.write(" ")
		fp.write("\n")
	f.close()
	fp.close()



# for i in range(1,3):
# 	label_name = "labels"+str(i)+".txt"
# 	path = os.path.join("train_labels","clean",label_name)
# 	f = open(path, "r")
# 	for x in f:
# 		x1 = x.split(" ")
# 		x1 = x1[0:5]
# 		x2 = [float(a) for a in x1]
# 		print(x2)
# 	f.close()