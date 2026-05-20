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


