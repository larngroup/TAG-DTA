import tensorflow as tf
import time
from dataset_builder_util import *
from layers_utils import *
from argument_parser import *


# Sparse categorical cross entropy loss value based on the masked input tokens
def loss_function(loss_object, real, preds, weights):
    loss_value = loss_object(real, preds, weights)

    return tf.reduce_sum(loss_value) / tf.reduce_sum(weights)


# Accuracy over the masked input tokens
def acc_function(real, preds, weights):
    acc = tf.equal(tf.cast(real, dtype=tf.int64), tf.cast(tf.argmax(preds, axis=2), dtype=tf.int64))
    mask = tf.math.logical_not(tf.math.equal(weights, 0))
    accuracies = tf.math.logical_and(mask, acc)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# Training and Testing/Validation Steps
def run_train_val(FLAGS, epochs, dataset_train, dataset_val, model, optimizer, loss_object, train_loss_object, train_acc_object,
                  val_loss_object, val_acc_object, checkpoint, checkpoint_manager, reset_checkpoint = True, es_num = 30):
    

    """
    Args:
    - FLAGS: arguments object
    - epochs[int]: number of epochs
    - dataset_train [TF Dataset]: [smiles_mlm_train_data, smiles_mlm_train_target, smiles_mlm_train_weights]
    - dataset_val [TF Dataset]: [smiles_mlm_val_data, smiles_mlm_val_target, smiles_mlm_val_weights]
    - model [TF Model]: BERT SMILES MLM Architecture
    - optimizer [TF Optimizer]: optimizer function
    - loss_object [TF Loss]: loss function
    - train_loss_object [TF Metric]: training loss object to compute loss over all batches at each epoch
    - train_acc_object [TF Metric]: training accuracy object to compute accuracy over all batches at each epoch
    - val_loss_object [TF Metric]: validation loss object to compute loss over all batches at each epoch
    - val_acc_object [TF Metric]: validation accuracy object to compute accuracy over all batches at each epoch
    - checkpoint [TF Train Checkpoint]: checkpoint object
    - checkpoint_manager [ TF Train Checkpoint Manager]: checkpoint object manager
    - reset_checkpoint [bool]: Train from scratch or start from a saved checkpoint
    - es_num [int]: early stopping patience parameter

    """

    if not reset_checkpoint:

        if checkpoint_manager.latest_checkpoint:

            print("Restored from {}".format(checkpoint_manager.latest_checkpoint))

            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            model = checkpoint.model
            optimizer = checkpoint.optimizer

        else:
            print("Start from scratch")

    else:
        print('Reset and Start from scratch')

    best_acc = 0.0
    es_count = 0
    best_epoch = 0
    
    @tf.function
    def train_step(x, y, weights, model, optimizer, loss_object, train_loss_object, train_acc_object):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_function(loss_object, y, logits, weights)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_loss_object(loss_value)
        train_acc_object(acc_function(y, logits, weights))

    @tf.function
    def test_step(x, y, weights, model, loss_object, val_loss_object, val_acc_object):
        val_logits = model(x, training=False)
        loss_value = loss_function(loss_object, y, val_logits, weights)
        val_loss_object(loss_value)
        val_acc_object(acc_function(y, val_logits, weights))

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        train_loss_object.reset_states()
        train_acc_object.reset_states()
        val_loss_object.reset_states()
        val_acc_object.reset_states()

        for batch, (x_batch_train, y_batch_train, weights_train) in enumerate(dataset_train):
            train_step(x_batch_train, y_batch_train, weights_train, model, optimizer, loss_object, train_loss_object, train_acc_object)

            if batch % 100 == 0:
            	print("Epoch : %d, Batch: %d, Train loss = %0.4f, Train acc = %0.4f" % (epoch, batch, float(train_loss_object.result()),float(train_acc_object.result())))

        for batch, (x_batch_val, y_batch_val, weights_val) in enumerate(dataset_val):
            test_step(x_batch_val, y_batch_val, weights_val, model, loss_object, val_loss_object, val_acc_object)


        print("Time taken: %0.2fs, Train loss: %0.4f, Train acc: %0.4f, Val loss: %0.4f, Val acc: %0.4f" % (time.time() - start_time,
                                                                                                          float(train_loss_object.result()),
                                                                                                        float(train_acc_object.result()),
                                                                                                        float(val_loss_object.result()),
                                                                                                       float(val_acc_object.result())))
                                                                                                       
        checkpoint.step.assign_add(1)
        es_count += 1
        if float(val_acc_object.result()) > best_acc and float(val_acc_object.result())-best_acc >=0.001:
            es_count = 0
            best_acc = float(val_acc_object.result())
            best_epoch = epoch

            checkpoint_manager.save(checkpoint_number = int(best_epoch))

            logging(("Epochs = %d, Train Loss = %0.4f, Train Acc = %0.4f, Val Loss = %0.4f, Val Acc = %0.4f" % (
            epoch, float(train_loss_object.result()),
            float(train_acc_object.result()),
            float(val_loss_object.result()),
            float(val_acc_object.result()))), FLAGS)

            print('Val Acc Improved at epoch: ', epoch)

        else:
            print('No improvement since epoch ', best_epoch)

        if es_count == es_num:
            break



def train_val(dataset,train_perc):
    """
    Hold-Out Training/Validation Split

    Args:
    - dataset [TF Dataset]: dataset in TF Dataset format
    - train_perc [int]: % Validation Ratio

    """
    dataset_shuffled = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=False)

    dataset_train = dataset_shuffled.take(int(train_perc*len(dataset_shuffled)))
    dataset_val = dataset_shuffled.skip(int(train_perc*len(dataset_shuffled)))

    return dataset_train, dataset_val
