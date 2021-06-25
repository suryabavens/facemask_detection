import pyrebase
from PIL import Image
from io import BytesIO
import re, time, base64

config = {
"apiKey": "************************",
"authDomain": "*************************",
"databaseURL": "*****************************",
"projectId": "***************",
"storageBucket": "*********************",
"messagingSenderId": "****************",
"appId": "******************",
"measurementId": "****************"

}

firebase = pyrebase.initialize_app(config)
i=0

auth = firebase.auth()
db = firebase.database()


def get_photo():
	all_users = db.child("photo").get()
	
	if all_users.each() is not None:
		for user in all_users.each():
			file1=""
			str1 = user.val()["vehiclenum2"]
			str2 = user.val()["today"]
			str3 = user.val()["photo"]
			str4 = str3[23:]
			file1 = 'input/' + str1 + "_" + str2 + ".png"
			str5 = base64.b64decode((str4))
			image_data = BytesIO(str5)
			img = Image.open(image_data)
			img.save(file1, "PNG")
			db.child("photo").child(user.key()).remove()
		return 1
	else:
		return 0
