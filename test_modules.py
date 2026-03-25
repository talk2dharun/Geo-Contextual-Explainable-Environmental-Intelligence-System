"""Test script for GEEIS modules."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_processing import prepare_data
from modules.ml_model import train_model, evaluate_model, get_feature_importance
from modules.knowledge_graph import build_knowledge_graph, get_graph_statistics

# Test data processing
print("Testing data processing...")
X_train, X_test, y_train, y_test, scaler, raw_df = prepare_data('data/water_potability.csv', 'models')
print("Data: X_train={}, X_test={}".format(X_train.shape, X_test.shape))
print("Classes: {}".format(y_train.value_counts().to_dict()))

# Test ML model
print("\nTraining ML model...")
model = train_model(X_train, y_train, 'models')
results = evaluate_model(model, X_test, y_test)
print("Accuracy: {:.4f}".format(results['accuracy']))
print("Precision: {:.4f}".format(results['precision']))
print("Recall: {:.4f}".format(results['recall']))
print("F1: {:.4f}".format(results['f1_score']))

# Test feature importance
imp = get_feature_importance(model)
print("\nTop features:")
print(imp.head())

# Test knowledge graph
print("\nBuilding knowledge graph...")
kg = build_knowledge_graph('data/Guidelines for drinking-water quality.pdf')
stats = get_graph_statistics(kg)
print("KG: {} nodes, {} edges".format(stats['total_nodes'], stats['total_edges']))
print("Pollutants: {}".format(stats['pollutants']))
print("Health impacts: {}".format(stats['health_impacts']))

print("\n=== All modules working! ===")
