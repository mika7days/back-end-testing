from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pyspark.sql.types import StructField, StructType, IntegerType
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

app = Flask(__name__)
CORS(app)

# initial_ratings = np.array([
#     [1, 1, 5],
#     [1, 2, 3],
#     [1, 3, 4],
#     [1, 4, 2],
#     [1, 5, None],
#     [2, 1, 3],
#     [2, 2, 1],
#     [2, 3, None],
#     [2, 4, 3],
#     [2, 5, 3],
#     [3, 1, 4],
#     [3, 2, None],
#     [3, 3, 4],
#     [3, 4, 3],
#     [3, 5, 5],
#     [4, 1, 3],
#     [4, 2, 3],
#     [4, 3, None],
#     [4, 4, 5],
#     [4, 5, 4],
#     [5, 1, None],
#     [5, 2, 5],
#     [5, 3, 5],
#     [5, 4, 2],
#     [5, 5, 1]
# ])

@app.route('/')
def index():
    ratings_dict = {}
    for row in initial_ratings:
        user_id, item_id, rating = row
        if user_id not in ratings_dict:
            ratings_dict[user_id] = {}
        ratings_dict[user_id][item_id] = rating
    return render_template('index.html', ratings_dict=ratings_dict)

@app.route('/update', methods=['POST'])
def update():
    updated_ratings = request.json.get('ratings')
    flat_ratings = []
    for user_id, items in updated_ratings.items():
        for item_id, rating in items.items():
            flat_ratings.append([int(user_id), int(item_id), rating])

    spark = SparkSession.builder.appName('Recommender').getOrCreate()
    schema = StructType([
        StructField('book_id', IntegerType(), True),
        StructField('user_id', IntegerType(), True),
        StructField('rating', IntegerType(), True)
    ])

    rdd = spark.sparkContext.parallelize(flat_ratings)
    df = spark.createDataFrame(rdd, schema)

    clean = df.na.drop()

    als = ALS(maxIter=5,
              regParam=0.01,
              rank=5,
              userCol="user_id",
              itemCol="book_id",
              ratingCol="rating",
              coldStartStrategy='drop',
              nonnegative=True)

    model = als.fit(clean)
    predicted_ratings = model.transform(df)
    user_item_ratings = predicted_ratings.toPandas().values
    spark.stop()

    users = set()
    items = set()

    ratings_dict = {}
    cold_start_problem = False
    for row in user_item_ratings:
        user_id, item_id, original_rating, rating = row

        users.add(user_id)
        items.add(item_id)

        if rating > 5.0:
            rating = 5.0
        if rating < 0:
            rating = 0.0

        if rating is None or np.isnan(rating):
            cold_start_problem = True

        if user_id not in ratings_dict:
            ratings_dict[user_id] = []

        ratings_dict[user_id].append(
            {int(item_id): round(rating, 2) if np.isnan(original_rating) else round(original_rating, 2)})

    if len(users) < 5 or len(items) < 5:
        cold_start_problem = True

    return jsonify(success=True, ratings=ratings_dict, cold_start_problem=cold_start_problem)

if __name__ == '__main__':
    app.run(debug=True)
