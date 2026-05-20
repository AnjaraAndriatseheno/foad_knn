from data_loader import load_normalized_data
from knn import Knn
 
 
normalized_data, labels, scaler = load_normalized_data("bienetre.csv", target_col="risque")
 
normalized_data = list(normalized_data)
labels          = list(labels)
 
 
total_size      = len(normalized_data)
end_of_train    = int(total_size * 0.70)
end_of_val      = int(total_size * 0.85)
 
training_data   = normalized_data[:end_of_train]
training_labels = labels[:end_of_train]
 
validation_data   = normalized_data[end_of_train:end_of_val]
validation_labels = labels[end_of_train:end_of_val]
 
test_data   = normalized_data[end_of_val:]
test_labels = labels[end_of_val:]
 
print(f"Training size   : {len(training_data)}")
print(f"Validation size : {len(validation_data)}")
print(f"Test size       : {len(test_data)}")
print("-" * 40)
 
model = Knn()
best_k, all_results = model.grid_search(
    training_data,   training_labels,
    validation_data, validation_labels,
    k_values_to_test=[1, 3, 5, 7, 9, 11]
)
 

final_accuracy = model.evaluate(test_data, test_labels)
print(f"\nFinal accuracy on test set : {final_accuracy:.4f}")