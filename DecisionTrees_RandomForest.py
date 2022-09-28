import numpy as np
import torch
from sklearn import tree, ensemble
from sklearn.datasets import fetch_openml
import math


# calculates number of correctly labeled samples and return accuracy
def evaluate(pred, lab):
    pred = torch.from_numpy(pred)

    n_correct = 0
    n_correct += (lab == pred).sum().item()

    return n_correct / len(pred) * 100


# finds tree with the best accuracy on validation data
def find_best_tree(trees_predictions, lab, trees):
    best_acc = 0
    best_tree_index = 0
    for index, pred in enumerate(trees_predictions):
        curr_acc = evaluate(pred, lab)

        if curr_acc > best_acc:
            best_tree_index = index
            best_acc = curr_acc

        print(f'Tree #: {index+1}, min_samples_leaves: {trees[index].min_samples_leaf}, criterion: {trees[index].criterion}, acc: {curr_acc: .4f}')


    return best_tree_index, best_acc


if __name__ == '__main__':
    print("Initiating fetch")
    inputs, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    print("Fetch completed")


    # convert imported data into tensors
    inputs = torch.from_numpy(inputs)

    # could also directly transform below list into tensor using torch.tensor(mylist)
    labels = np.array([int(x) for x in labels])
    labels = torch.from_numpy(labels)


    # functional variables
    trees_in_forest = 40

    tot_examples = inputs.shape[0]  # same as len(inputs)

    # Note that validation data are later useful in order to determine the best tree features (ex. depth, criterion (gini vs entropy), ...)
    tot_train_examples = int(math.ceil(tot_examples)*5/7)
    tot_val_examples = int(math.ceil((tot_examples - tot_train_examples)/2))
    tot_test_examples = tot_examples - tot_train_examples - tot_val_examples

    train_val_split = (tot_train_examples, tot_val_examples, tot_test_examples)


    train_dataset_in, val_dataset_in, test_dataset_in = inputs.split(train_val_split)
    train_dataset_out, val_dataset_out, test_dataset_out = labels.split(train_val_split)


    #training

    # defining multiple models (for classification) using list comprehension
    # note that defining the number of minimum leaves is important because we want to avoid overfitting;
    # a model would normally reach n_samples_leaf = 1 for each leaf if trained too much on training samples, but failing to generalize
    print("Building trees")
    dtrees = [tree.DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_leaf).fit(train_dataset_in, train_dataset_out)for min_leaf in range(1, 4) for criterion in ["gini", "entropy"]]
    print("Finished building trees")

    print("Finding best tree\n")
    predictions = [tr.predict(val_dataset_in) for tr in dtrees]

    best_tree, best_tree_acc = find_best_tree(predictions, val_dataset_out, dtrees)

    print(f'\nSelected tree #: {best_tree+1} with an accuracy on validation data of: {best_tree_acc:.4f}')


    # testing

    test_pred = dtrees[best_tree].predict(test_dataset_in)

    test_acc = evaluate(test_pred, test_dataset_out)
    
    print(f'Accuracy on test data: {test_acc:.4f}')


    # RandomForest
    # the model uses a criterion the one of the best_tree previously selected
    rforest = ensemble.RandomForestClassifier(n_estimators=trees_in_forest, criterion=dtrees[best_tree].criterion)

    print("\nCreating random forest")
    rforest.fit(train_dataset_in, train_dataset_out)
    print("Finished creating forest")

    # we haven't used validation data during the training process, so we can use them as test data
    prediction = rforest.predict(torch.cat((val_dataset_in, test_dataset_in)))

    forest_acc = evaluate(prediction, torch.cat((val_dataset_out, test_dataset_out)))

    print(f'\nAccuracy of random forest with {rforest.n_estimators} estimators and criterion "{rforest.criterion}": {forest_acc:.4f}')

