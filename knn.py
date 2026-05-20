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
    
    def evaluate(self, test_data, real_labels):
        predictions   = self.predict(test_data)
        correct_count = 0
        for prediction, real_label in zip(predictions, real_labels):
            if prediction == real_label:
                correct_count += 1
        accuracy = correct_count / len(real_labels)
        return accuracy
    
    def grid_search(self, training_data, training_labels,
                    validation_data, validation_labels,
                    k_values_to_test):
        best_k        = None
        best_accuracy = -1
        all_results   = {}

        for k_candidate in k_values_to_test:
            self.neighbors_number = k_candidate
            self.fit(training_data, training_labels)

            validation_accuracy      = self.evaluate(validation_data, validation_labels)
            all_results[k_candidate] = validation_accuracy
            print(f"k={k_candidate} → validation accuracy = {validation_accuracy:.4f}")

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_k        = k_candidate
            
        self.neighbors_number = best_k
        self.fit(training_data, training_labels)
        print(f"\nBest k : {best_k} (accuracy = {best_accuracy:.4f})")
        return best_k, all_results