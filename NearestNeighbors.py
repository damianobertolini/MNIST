import torch
from sklearn import neighbors
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def evaluate(pred, label):
    pred = torch.from_numpy(pred)

    n_correct = 0

    wrong_indexes = []

    for ind, (p, l) in enumerate(zip(pred, label)):
        if p == l:
            n_correct += 1
        else:
            wrong_indexes.append(ind)

    return n_correct / len(pred) * 100, wrong_indexes


def find_best_knn_model(preds, label, models):
    best_acc_index = 0
    best_acc = -1
    wrong_ind = []

    for index, pred in enumerate(preds):
        curr_acc, wrong_ind = evaluate(pred, label)

        if curr_acc > best_acc:
            best_acc = curr_acc
            best_acc_index = index

        print(f'Model #: {index+1}, weight type: "{models[index].weights}", n_neighbors: {models[index].n_neighbors}, acc: {curr_acc: .4f}')


    return best_acc, wrong_ind, best_acc_index

if __name__ == '__main__':
    print("Initiating fetch")
    inputs, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    print("Fetch completed")

    inputs = torch.from_numpy(inputs)

    # could also directly transform below list into tensor using torch.tensor(mylist)
    labels = torch.tensor([int(x) for x in labels])

    # variables
    k = [x for x in range(1, 10)]

    total_examples = len(inputs)
    train_examples = int(len(inputs) * 6 / 7)
    test_examples = total_examples - train_examples
    train_test_split = (train_examples, test_examples)

    # note that the split is not random, so data will be aligned
    train_in, test_in = inputs.split(train_test_split)
    train_out, test_out = labels.split(train_test_split)

    # defining model
    knns = [neighbors.KNeighborsClassifier(n_jobs=-1, weights=weight, n_neighbors=n_neighbor) for weight in ["uniform", "distance"] for n_neighbor in range(1, 5)]

    # training
    print("Initiating training")
    [knn.fit(train_in, train_out) for knn in knns]
    print("Finished training")

    # testing
    print("Finding best model\n")
    test_preds = [knn.predict(test_in) for knn in knns]

    best_test_acc, best_wrong_ind, best_model_index = find_best_knn_model(test_preds, test_out, knns)

    print(f'\nSelected model #: {best_model_index+1} with {knns[best_model_index].n_neighbors} neighbors and weight type: "{knns[best_model_index].weights}" in prediction process and accuracy on testing data of: {best_test_acc:.4f}')

    while True:
        n_to_show = int(input(f'\nInsert how many of {len(best_wrong_ind)} wrongly classified digits of the best model to show: '))

        if 0 < n_to_show <= len(best_wrong_ind):
            break
        else:
            print(f'Insert value between 1 and {len(best_wrong_ind)}')

    # prints all wrongly classified digits with the predicted label as title
    # change title font size
    par = {'axes.titlesize': 20}
    plt.rcParams.update(par)

    for j in range(0, n_to_show, 6):
        if j + 6 > n_to_show:
            n = n_to_show - j
        else:
            n = 6
        for i in range(n):
            if j + i < len(best_wrong_ind):
                img_plt = test_in[best_wrong_ind[j + i]].reshape(28, 28)
                plt.subplot(2, 3, i + 1)
                plt.imshow(img_plt, cmap='gray')
                plt.title(f'pred: {test_preds[best_model_index][best_wrong_ind[j + i]]}')

        plt.show()
