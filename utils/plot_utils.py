import matplotlib.pyplot as plt

def plot_training_curve(losses, accuracies):
    fig, ax1 = plt.subplots()
    ax1.plot(losses, 'r-', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='r')
    ax2 = ax1.twinx()
    ax2.plot(accuracies, 'b-', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='b')
    plt.title('Training Loss and Accuracy')
    plt.show()
