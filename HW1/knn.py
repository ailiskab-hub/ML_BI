import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        n_train = self.train_X.shape[0]
        n_test = X.shape[0]
        dist = np.zeros((n_test, n_train))
       
        for i in range(n_test):
            for j in range(n_train):
                dist[i, j] = np.sum(np.abs(X[i] - self.train_X[j]))
        return dist

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        n_train = self.train_X.shape[0]
        n_test = X.shape[0]
        dist = np.zeros((n_test, n_train))
        for i in range(n_test):
            abs_dif = np.abs(X[i] - self.train_X)
            dist[i] = np.sum(abs_dif, axis=1)
        return dist


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        abs_dif = np.abs(X[:, np.newaxis] - self.train_X)
        dist = np.sum(abs_dif, axis=-1)

        return dist


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        #n_train = distances.shape[]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        for i in range(n_test):
            prob_y = np.argsort(distances[i])
            prob_lab = self.train_y[prob_y[:self.n_neighbors]]
            prediction[i] = float(float(np.sum(prob_lab)) >= self.n_neighbors / 2)

        return prediction
    
    
    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        for i in range(n_test):
            nn_ind = np.argmin(distances[i])
            prediction[i] = self.train_y[nn_ind]
        return prediction