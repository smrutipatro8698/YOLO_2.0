import os

train_path = os.path.join("data","custom","train.txt")
f = open(train_path,"w")
for i in range(1,150):
	image_name = "image"+str(i)+".png"
	path_name = "data/"+"custom/"+"images/"+image_name
	# path_name = os.path.join("data","custom","images",image_name)
	# path_name1 = os.path.normpath(path_name)
	f.write(path_name)
	f.write("\n")
f.close()

valid_path = os.path.join("data","custom","valid.txt")
f1 = open(valid_path,"w")
for i in range(150,184):
	image_name = "image"+str(i)+".png"
	path_name = "data/"+"custom/"+"images/"+image_name
	#path_name = os.path.join(os.sep,"data","custom","images",image_name)
	f1.write(path_name)
	f1.write("\n")
f1.close()
