class Knn : 
    def __init__(self, neighbors_number=3):
        self.neighbors_number = neighbors_number
        self.donnees_entrainement = None           
        self.etiquettes = None

    
    def read(self, training_data, training_labels):
        self.training_data   = training_data
        self.training_labels = training_labels
    

    def _compute_distance(self, point_a, point_b):
        sum_of_squares = 0
        for index in range(len(point_a)):
            difference      = point_a[index] - point_b[index]
            sum_of_squares += difference ** 2
        return sum_of_squares ** 0.5


    def _predict_one_point(self, unknown_point): 
        distances_list = []
        for index, known_point in enumerate(self.training_data):
            distance = self._compute_distance(unknown_point, known_point)
            distances_list.append((distance, self.training_labels[index]))
            
        distances_list.sort(key=lambda pair: pair[0])
        nearest_neighbors = distances_list[:self.neighbors_number]

        votes_counter = {}
        for distance, label in nearest_neighbors:
            votes_counter[label] = votes_counter.get(label, 0) + 1
            
        winning_class = max(votes_counter, key=votes_counter.get)
        return winning_class