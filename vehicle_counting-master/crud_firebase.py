from firebase import firebase
import psycopg2
import datetime


date_time = datetime.datetime.now()

firebase = firebase.FirebaseApplication("https://vehicle-1d2dd.firebaseio.com/", None)

data = {
	"date_time": str(date_time),
	"vehicle_type": "car",
	"confidence": 987654321,
    "weight": 100
}

result = firebase.post("/vehicle-1d2dd/Costumer", data)
print(result)


# try:
#     connection = psycopg2.connect(user = "postgres",
#                                   password = "12345678",
#                                   host = "127.0.0.1",
#                                   port = "5432",
#                                   database = "detection_vehicle")
#
#     cursor = connection.cursor()
#     #Print PostgreSQL Connection properties
#     print ( connection.get_dsn_parameters(),"\n")
#
#     # Print PostgreSQL version
#     cursor.execute("SELECT version();")
#     record = cursor.fetchone()
#     print("You are connected to - ", record,"\n")
#
# except (Exception, psycopg2.Error) as error :
#     print ("Error while connecting to PostgreSQL", error)



#postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence_, weight_) VALUES (%s,%s,%s,%s)"""
#record_to_insert = (str(date_time), "car", 0.93483, 100)
#cursor.execute(postgres_insert_query, record_to_insert)
#connection.commit()