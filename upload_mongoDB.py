import pandas as pd
import pymongo


df = pd.read_csv(r'D:\FFR-AI\Working\Working\recommender-system\students_marks\student_scores.csv')
print (df)
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client["student_marks"]
collection = db['Student_scores']

# Convert DataFrame to dictionary and upload to MongoDB
data_dict = df.to_dict(orient='records')
collection.insert_many(data_dict)
print("Data uploaded to MongoDB successfully.")