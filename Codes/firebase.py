
import pyrebase

# import pyqtgraph as pg
# from firebase import firebase

config={
   "apiKey": "AIzaSyCSI1XHYpvzVHFawjEKMlJnlrjJTnN6YdQ",
   "authDomain": "vital-e5b32.firebaseapp.com",
   "databaseURL": "https://vital-e5b32-default-rtdb.firebaseio.com/",
   "storageBucket": "vital-e5b32.appspot.com"

}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
data={
"Project Name":"Vital sign detection"
"Project Members":{
"D K Chandrakanth": "1SI20EC020",
"Preetham G M":"1SI20EC121",
"Nithesh Kumar V":"1SI20EC122",
"Mithun J B":"1SI20EC124"}
}
data1={
"Name": "Name",
"Age": "Age",
"Gender":"Gender",
"ID": "ID"
}
db.child("mlx90614").child("1-set").set(data)
db.child("mlx90614").child("2-push").push(data1)
