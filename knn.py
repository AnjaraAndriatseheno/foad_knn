class Knn : 
    def __init__(self, neighbors_number=3):
        self.neighbors_number = neighbors_number
        self.donnees_entrainement = None           
        self.etiquettes = None
    
    def read(self, training_data, training_labels):
        self.training_data   = training_data
        self.training_labels = training_labels


