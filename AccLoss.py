import matplotlib.pyplot as plt


def acc_loss(model, font):
    acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(7, 7))
    plt.plot(epochs, acc, 'bo', label='Train accuracy', linewidth=0.7)
    plt.plot(epochs, val_acc, 'b', label='Test accuracy')
    plt.xlabel('Epochs', font)
    plt.ylabel('Accuracy', font)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower right')
    plt.show()
    # loss
    plt.figure(figsize=(7, 7))
    plt.plot(epochs, model.history.history['loss'])
    plt.plot(epochs, model.history.history['val_loss'])
    plt.ylabel('Loss', font)
    plt.xlabel('Epoch', font)
    plt.tick_params(labelsize=14)
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

