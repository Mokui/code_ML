import matplotlib.pyplot as plt
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import train_test_split

#use surprise

data = Dataset.load_builtin('ml-100k')

# split the data with test-train-split
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# SVD algorithm because its the same as Netflix, which is cool because we use the movielens package
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# we need predictions r_ui and est and make delta
deltas = [ elem[2]- elem[3] for elem in predictions ]

# On plot l'histogramme
plt.hist(deltas, 50, color = 'gray', edgecolor = 'black')

plt.show()
