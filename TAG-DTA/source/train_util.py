import tensorflow as tf
from sklearn.metrics import r2_score as r2s
from scipy import stats
import time
from args_parser import *
from main import *
import math
from utils import *
from layers_utils import *

# Step decay scheduler based on the training cycles
def step_decay_lr(init_lr, drop_rate, current_epoch, epochs_drop):
    new_lr = init_lr * math.pow(drop_rate, math.floor(current_epoch / epochs_drop))
    return new_lr

# Weighted Binary Cross Entropy or Binary Focal Cross Entropy loss value
def bind_loss_function(loss_object, real, preds, weights, loss_opt):
    ones = tf.cast(tf.math.logical_and(tf.math.equal(real, 1), tf.equal(weights, 1)), tf.float32)
    zeros = tf.cast(tf.math.logical_and(tf.math.equal(real, 0), tf.equal(weights, 1)), tf.float32)

    score_ones = tf.reduce_sum(ones) / (tf.reduce_sum(ones) + tf.reduce_sum(zeros))

    if loss_opt[0] == 'focal' and loss_opt[2] == '' and loss_opt[3] == '':
        weights = ones + (1 - score_ones) + zeros * score_ones

    elif loss_opt[0] == 'focal':
        weights = zeros * float(loss_opt[2]) + ones * float(loss_opt[3])

    elif loss_opt[0] == 'standard' and loss_opt[1] == '' and loss_opt[2] == '':
        weights = ones * (1 - score_ones) + zeros * score_ones

    elif loss_opt[0] == 'standard':
        weights = zeros * float(loss_opt[1]) + ones * float(loss_opt[2])

    # weights = ones * 0.60 + zeros * 0.40

    loss_value = loss_object(real, preds, weights)

    loss_value = tf.reduce_sum(loss_value) / tf.reduce_sum(weights)

    return loss_value

# Binding Pocket Evaluation metrics
def bind_metrics_function(real, preds, weights, threshold=0.5):
    real = tf.cast(real, dtype=tf.int64)
    preds = tf.cast(tf.nn.sigmoid(preds) > threshold, tf.int64)
    # preds = tf.cast(tf.argmax(preds, axis=2), dtype=tf.int64)
    weights = tf.math.logical_not(tf.math.equal(weights, 0))

    tp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 1), tf.equal(preds, 1)), weights),
                               dtype=tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 0), tf.equal(preds, 0)), weights),
                               dtype=tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 0), tf.equal(preds, 1)), weights),
                               dtype=tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_and(tf.equal(real, 1), tf.equal(preds, 0)), weights),
                               dtype=tf.float32))

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-08)
    recall = tp / (tp + fn + 1e-08)
    bacc = (recall + (tn / (tn + fp + 1e-08))) * 0.5
    precision = tp / (tp + fp + 1e-08)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-08)
    mcc = ((tp * tn) - (fp * fn)) / tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-08)

    return bacc, recall, precision, f1, mcc




# Concordance Index Function
def c_index(y_true, y_pred):
    matrix_pred = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    matrix_pred = tf.cast(matrix_pred == 0.0, tf.float32) * 0.5 + tf.cast(matrix_pred > 0.0, tf.float32)

    matrix_true = tf.subtract(tf.expand_dims(y_true, -1), y_true)
    matrix_true = tf.cast(matrix_true > 0.0, tf.float32)

    matrix_true_position = tf.where(tf.equal(matrix_true, 1))

    matrix_pred_values = tf.gather_nd(matrix_pred, matrix_true_position)

    # If equal to zero then it returns zero, else return the result of the division
    result = tf.where(tf.equal(tf.reduce_sum(matrix_pred_values), 0), 0.0,
                      tf.reduce_sum(matrix_pred_values) / tf.reduce_sum(matrix_true))

    return result


# Get Pre-Trained SMILES Transformer-Encoder, Binding Affinity Layers, and Binding Pocket Layers Trainable Variables
def trainable_variables(model):
    smiles_bert_variables = model.get_layer('smiles_bert').trainable_variables
    bind_vector_variables = []
    for i in model.layers:
        if i._name != 'smiles_bert' and i._name.split('_')[0] != 'affinity':
            bind_vector_variables += i.trainable_variables

    affinity_variables = []
    for i in model.layers:
        if i._name != 'smiles_bert' and i._name.split('_')[0] != 'binding':
            affinity_variables += i.trainable_variables

    return smiles_bert_variables, bind_vector_variables, affinity_variables

# Training Mode: Binding Affinity Prediction or Binding Pocket Prediction
def training_mode(model, option):
    if option == 'binding':
        for i in model.layers:
            if i._name.split('_')[0] == 'affinity':
                model.get_layer(i._name).trainable = False

            if i._name.split('_')[0] == 'binding':
                model.get_layer(i._name).trainable = True

    if option == 'affinity':
        for i in model.layers:
            if i._name.split('_')[0] == 'affinity':
                model.get_layer(i._name).trainable = True

            if i._name.split('_')[0] == 'binding':
                model.get_layer(i._name).trainable = False

# Training and Testing/Validation Steps
def run_train_val(FLAGS, fold_idx, epochs, bind_data_train, bind_data_val, affinity_data_train, affinity_data_val,
                  model, smiles_opt, bind_opt, affinity_opt, bind_loss_opt, bind_loss_obj, bind_train_loss_obj,
                  bind_train_acc_obj, bind_train_recall_obj, bind_train_precision_obj, bind_train_f1_obj,
                  bind_train_mcc_obj, bind_val_loss_obj, bind_val_acc_obj, bind_val_recall_obj, bind_val_precision_obj,
                  bind_val_f1_obj, bind_val_mcc_obj, affinity_loss_fn, affinity_train_loss_obj, affinity_train_rmse_obj,
                  affinity_train_ci_obj, affinity_val_loss_obj, affinity_val_rmse_obj, affinity_val_ci_obj,
                  ckpt, ckpt_manager, pre_train_epochs, bind_vector_epochs, bind_affinity_epochs,
                  es_value=25, affinity_drop_train_cycle=25, bind_vector_drop_train_cycle=25, lr_drop_rate=0.5,
                  reset_checkpoint=False):


    """
    Args:
    - FLAGS: arguments object
    - fold_idx [int]: affinity cluster index
    - epochs[int]: number of epochs
    - bind_data_train [TF Dataset]: [binding_train_protein_data, binding_train_smiles_data, binding_train_target, binding_train_weights]
    - bind_data_val [TF Dataset]: [binding_val_protein_data, binding_val_smiles_data, binding_val_target, binding_val_weights]
    - affinity_data_train [TF Dataset]: [affinity_train_protein_data, affinity_train_smiles_data, affinity_train_target]
    - affinity_data_val [TF Dataset]: [affinity_val_protein_data, affinity_val_smiles_data, affinity_val_target]
    - model [TF Model]: TAG-DTA Model Architecture
    - smiles_opt [TF Optimizer]: Pre-Trained SMILES Transformer-Encoder optimizer function
    - bind_opt [TF Optimizer]: 1D Binding Pocket Classifier optimizer function
    - affinity_opt [TF Optimizer]: Binding Affinity Regressor optimizer function
    - bind_loss_opt [string]: binding pocket loss function option and class weights
    - bind_loss_obj [TF Loss]: binding pocket loss function
    - bind_train_loss_obj [TF Metric]: binding pocket training loss object to compute loss over all batches at each epoch
    - bind_train_acc_obj [TF Metric]: binding pocket training accuracy object to compute accuracy over all batches at each epoch
    - bind_train_recall_obj [TF Metric]: binding pocket training recall object to compute loss over all batches at each epoch
    - bind_train_precision_obj [TF Metric]: binding pocket training precision object to compute accuracy over all batches at each epoch
    - bind_train_f1_obj [TF Metric]: binding pocket training f1-score object to compute accuracy over all batches at each epoch
    - bind_train_mcc_obj [TF Metric]: binding pocket training mcc object to compute accuracy over all batches at each epoch
    - bind_val_loss_obj [TF Metric]: binding pocket validation loss object to compute loss over all batches at each epoch
    - bind_val_acc_obj [TF Metric]: binding pocket validation accuracy object to compute accuracy over all batches at each epoch
    - bind_val_recall_obj [TF Metric]: binding pocket validation recall object to compute loss over all batches at each epoch
    - bind_val_precision_obj [TF Metric]: binding pocket validation precision object to compute accuracy over all batches at each epoch
    - bind_val_f1_obj [TF Metric]: binding pocket validation f1-score object to compute accuracy over all batches at each epoch
    - bind_val_mcc_obj [TF Metric]: binding pocket validation mcc object to compute accuracy over all batches at each epoch
    - affinity_loss_fn [TF Loss]: binding affinity loss function
    - affinity_train_loss_obj [TF Metric]: binding affinity training loss object to compute loss over all batches at each epoch
    - affinity_train_rmse_obj [TF Metric]: binding affinity training rmse object to compute loss over all batches at each epoch
    - affinity_train_ci_obj [TF Metric]: binding affinity training ci object to compute loss over all batches at each epoch
    - affinity_val_loss_obj [TF Metric]: binding affinity validation loss object to compute loss over all batches at each epoch
    - affinity_val_rmse_obj [TF Metric]:binding affinity validation rmse object to compute loss over all batches at each epoch
    - affinity_val_ci_obj [TF Metric]: binding affinity validation ci object to compute loss over all batches at each epoch
    - ckpt [TF Train Checkpoint]: checkpoint object
    - ckpt_manager [ TF Train Checkpoint Manager]: checkpoint object manager
    - pre_train_epochs [int]: number of binding pocket pre-training epochs
    - bind_vector_epochs [int]: number of binding pocket epochs
    - bind_affinity_epochs [int]: number of binding affinity epochs
    - es_value [int]: early stopping patience parameter based on training cycles
    - affinity_drop_train_cycle [int]: number of training cycles that the learning rate keeps constant for the binding affinity optimizer 
    - bind_vector_drop_train_cycle [int]: number of training cycles that the learning rate keeps constant for the binding pocket optimizer 
    - lr_drop_rate [float]: step decay learning rate dropping factor
    - reset_checkpoint [bool]: Train from scratch or start from a saved checkpoint

    """


    if not reset_checkpoint:
        if ckpt_manager.latest_checkpoint:
            print('Restore from {}'.format(ckpt_manager.latest_checkpoint))

        else:
            print('Start from scratch')

    else:
        print('Reset and start from scratch')

    @tf.function
    def bind_train_step(x1, x2, y, weights, bind_loss_opt, model, smiles_opt, binding_opt, bind_loss_obj,
                        bind_train_loss_obj, bind_train_acc_obj, bind_train_recall_obj,
                        bind_train_precision_obj, bind_train_f1_obj, bind_train_mcc_obj):


        smiles_variables, bind_variables, _ = trainable_variables(model)
        with tf.GradientTape() as tape:
            logits = model([x1, x2], training=True)
            loss_value = bind_loss_function(bind_loss_obj, y, logits[0], weights, bind_loss_opt)

        train_variables = smiles_variables + bind_variables
        gradients = tape.gradient(loss_value, train_variables)

        smiles_grads = gradients[:len(smiles_variables)]
        bind_grads = gradients[len(smiles_variables):len(smiles_variables) + len(bind_variables)]

        smiles_opt.apply_gradients(zip(smiles_grads, smiles_variables))
        binding_opt.apply_gradients(zip(bind_grads, bind_variables))

        bind_train_loss_obj(loss_value)
        bacc, recall, precision, f1, mcc = bind_metrics_function(y, logits[0], weights)
        bind_train_acc_obj(bacc)
        bind_train_recall_obj(recall)
        bind_train_precision_obj(precision)
        bind_train_f1_obj(f1)
        bind_train_mcc_obj(mcc)

    @tf.function
    def affinity_train_step(x1, x2, y, model, affinity_loss_fn, smiles_opt, affinity_opt, affinity_train_loss_obj,
                            affinity_train_rmse_obj, affinity_train_ci_obj):

        smiles_bert_variables, _, affinity_variables = trainable_variables(model)
        with tf.GradientTape() as tape:
            logits = model([x1, x2], training=True)
            loss_value = affinity_loss_fn(y, logits[1])

        train_variables = smiles_bert_variables + affinity_variables
        gradients = tape.gradient(loss_value, train_variables)

        smiles_grads = gradients[:len(smiles_bert_variables)]
        affinity_grads = gradients[len(smiles_bert_variables):len(smiles_bert_variables) + len(affinity_variables)]

        smiles_opt.apply_gradients(zip(smiles_grads, smiles_bert_variables))
        affinity_opt.apply_gradients(zip(affinity_grads, affinity_variables))

        affinity_train_loss_obj(loss_value)
        affinity_train_rmse_obj(rmse_function(y, logits[1]))
        affinity_train_ci_obj(c_index(y, logits[1]))

    @tf.function
    def bind_test_step(x1, x2, y, weights, model, bind_loss_opt, bind_loss_obj, bind_val_loss_obj,
                       bind_val_acc_obj, bind_val_recall_obj, bind_val_precision_obj, bind_val_f1_obj,
                       bind_val_mcc_obj):

        logits = model([x1, x2], training=False)
        loss_value = bind_loss_function(bind_loss_obj, y, logits[0], weights, bind_loss_opt)
        bind_val_loss_obj(loss_value)
        bacc, recall, precision, f1, mcc = bind_metrics_function(y, logits[0], weights)
        bind_val_acc_obj(bacc)
        bind_val_recall_obj(recall)
        bind_val_precision_obj(precision)
        bind_val_f1_obj(f1)
        bind_val_mcc_obj(mcc)

    @tf.function
    def affinity_test_step(x1, x2, y, model, affinity_loss_fn, affinity_val_loss_obj, affinity_val_rmse_obj,
                           affinity_val_ci_obj):


        logits = model([x1, x2], training=False)
        loss_value = affinity_loss_fn(y, logits[1])
        affinity_val_loss_obj(loss_value)
        affinity_val_rmse_obj(rmse_function(y, logits[1]))
        affinity_val_ci_obj(c_index(y, logits[1]))


    best_epoch = 0
    best_bind_metric = -1.0
    best_affinity_metric = 1000
    es_count = 0
    bind_ratio = 0
    affinity_ratio = 0
    train_cycle = 0

    init_affinity_lr = affinity_opt.lr.numpy()
    init_bind_vector_lr = bind_opt.lr.numpy()

    for epoch in range(epochs):
        print("Epoch: {}".format(str(epoch)))
        start_time = time.time()

        new_affinity_lr = step_decay_lr(init_affinity_lr, lr_drop_rate, train_cycle, affinity_drop_train_cycle)
        new_bind_lr = step_decay_lr(init_bind_vector_lr, lr_drop_rate, train_cycle, bind_vector_drop_train_cycle)

        bind_opt.lr.assign(new_bind_lr)
        affinity_opt.lr.assign(new_affinity_lr)

        if epoch < pre_train_epochs:
            training_mode(model, 'binding')
            bind_train_loss_obj.reset_states()
            bind_train_acc_obj.reset_states()
            bind_train_recall_obj.reset_states()
            bind_train_precision_obj.reset_states()
            bind_train_f1_obj.reset_states()
            bind_train_mcc_obj.reset_states()

            bind_val_loss_obj.reset_states()
            bind_val_acc_obj.reset_states()
            bind_val_recall_obj.reset_states()
            bind_val_precision_obj.reset_states()
            bind_val_f1_obj.reset_states()
            bind_val_mcc_obj.reset_states()

            print('---------------Binding Vector Prediction Pre-Training---------------')
            for step, (prot_input, smiles_input, target, weights) in enumerate(bind_data_train):
                bind_train_step(prot_input, smiles_input, target, weights, bind_loss_opt,
                                model, smiles_opt, bind_opt,
                                bind_loss_obj, bind_train_loss_obj, bind_train_acc_obj, bind_train_recall_obj,
                                bind_train_precision_obj, bind_train_f1_obj, bind_train_mcc_obj)

                if step % 1000 == 0:
                    print((
                                  "Binding Vector Pre-Train Epoch: %d, Step: %d, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, "
                                  "Train Precision: %0.4f, " +
                                  "Train F1: %0.4f, Train MCC: %0.4f") % (
                              epoch, step, float(bind_train_loss_obj.result()), float(bind_train_acc_obj.result()),
                              float(bind_train_recall_obj.result()), float(bind_train_precision_obj.result()),
                              float(bind_train_f1_obj.result()), float(bind_train_mcc_obj.result())))

            for step, (prot_input, smiles_input, target, weights) in enumerate(bind_data_val):
                bind_test_step(prot_input, smiles_input, target, weights, model, bind_loss_opt,
                               bind_loss_obj, bind_val_loss_obj, bind_val_acc_obj, bind_val_recall_obj,
                               bind_val_precision_obj,
                               bind_val_f1_obj, bind_val_mcc_obj)

            end_time = time.time() - start_time

            print("------------------------//------------------------")
            print((
                          "Binding Vector Pre-Train Epoch: %d, Time Taken: %0.2f, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, Train Precision: %0.4f, " +
                          "Train F1: %0.4f, Train MCC: %0.4f, Val Loss: %0.4f, Val Acc: %0.4f, Val Recall: %0.4f, Val Precision: %0.4f, " +
                          "Val F1: %0.4f, Val MCC: %0.4f") %
                  (epoch, end_time, float(bind_train_loss_obj.result()), float(bind_train_acc_obj.result()),
                   float(bind_train_recall_obj.result()), float(bind_train_precision_obj.result()),
                   float(bind_train_f1_obj.result()), float(bind_train_mcc_obj.result()),
                   float(bind_val_loss_obj.result()), float(bind_val_acc_obj.result()),
                   float(bind_val_recall_obj.result()), float(bind_val_precision_obj.result()),
                   float(bind_val_f1_obj.result()), float(bind_val_mcc_obj.result())))
            print("------------------------//------------------------")
            ckpt.step.assign_add(1)
            ckpt_manager.save(checkpoint_number=epoch)

        if epoch >= pre_train_epochs and affinity_ratio < bind_affinity_epochs:
            training_mode(model, 'affinity')
            affinity_train_loss_obj.reset_states()
            affinity_train_rmse_obj.reset_states()
            affinity_train_ci_obj.reset_states()

            affinity_val_loss_obj.reset_states()
            affinity_val_rmse_obj.reset_states()
            affinity_val_ci_obj.reset_states()

            print('---------------Binding Affinity Prediction Training---------------')
            for step, (prot_input, smiles_input, kd_values) in enumerate(affinity_data_train):
                affinity_train_step(prot_input, smiles_input, kd_values, model, affinity_loss_fn, smiles_opt,
                                    affinity_opt, affinity_train_loss_obj, affinity_train_rmse_obj,
                                    affinity_train_ci_obj)

                if step % 1000 == 0:
                    print("Epoch: %d, Step: %d, Train Loss: %0.4f, Train RMSE: %0.4f, Train CI: %0.4f" %
                          (epoch, step, float(affinity_train_loss_obj.result()),
                           float(affinity_train_rmse_obj.result()),
                           float(affinity_train_ci_obj.result())))

            for step, (prot_input, smiles_input, kd_values) in enumerate(affinity_data_val):
                affinity_test_step(prot_input, smiles_input, kd_values, model, affinity_loss_fn,
                                   affinity_val_loss_obj, affinity_val_rmse_obj, affinity_val_ci_obj)

            end_time = time.time() - start_time
            ckpt.step.assign_add(1)

            print("------------------------//------------------------")
            print((
                          "Binding Affinity Epoch: %d, Time Taken: %0.2f, Fold: %d, Train Loss: %0.4f, Train RMSE: %0.4f, Train CI: %0.4f, " +
                          "Val Loss: %0.4f, Val RMSE: %0.4f, Val CI: %0.4f") % (
                      epoch, end_time, fold_idx, float(affinity_train_loss_obj.result()),
                      float(affinity_train_rmse_obj.result()),
                      float(affinity_train_ci_obj.result()),
                      float(affinity_val_loss_obj.result()),
                      float(affinity_val_rmse_obj.result()),
                      float(affinity_val_ci_obj.result())))
            print("------------------------//------------------------")
            affinity_ratio += 1

        if epoch >= pre_train_epochs and affinity_ratio >= bind_affinity_epochs:
            training_mode(model, 'binding')
            bind_train_loss_obj.reset_states()
            bind_train_acc_obj.reset_states()
            bind_train_recall_obj.reset_states()
            bind_train_precision_obj.reset_states()
            bind_train_f1_obj.reset_states()
            bind_train_mcc_obj.reset_states()

            bind_val_loss_obj.reset_states()
            bind_val_acc_obj.reset_states()
            bind_val_recall_obj.reset_states()
            bind_val_precision_obj.reset_states()
            bind_val_f1_obj.reset_states()
            bind_val_mcc_obj.reset_states()

            print('---------------Binding Vector Prediction Training---------------')
            for step, (prot_input, smiles_input, target, weights) in enumerate(bind_data_train):
                bind_train_step(prot_input, smiles_input, target, weights, bind_loss_opt,
                                model, smiles_opt, bind_opt,
                                bind_loss_obj, bind_train_loss_obj, bind_train_acc_obj, bind_train_recall_obj,
                                bind_train_precision_obj, bind_train_f1_obj, bind_train_mcc_obj)

                if step % 1000 == 0:
                    print((
                                  "Binding Vector Epoch: %d, Step: %d, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, "
                                  "Train Precision: %0.4f, " +
                                  "Train F1: %0.4f, Train MCC: %0.4f") % (
                              epoch, step, float(bind_train_loss_obj.result()), float(bind_train_acc_obj.result()),
                              float(bind_train_recall_obj.result()), float(bind_train_precision_obj.result()),
                              float(bind_train_f1_obj.result()), float(bind_train_mcc_obj.result())))

            for step, (prot_input, smiles_input, target, weights) in enumerate(bind_data_val):
                bind_test_step(prot_input, smiles_input, target, weights, model, bind_loss_opt,
                               bind_loss_obj, bind_val_loss_obj, bind_val_acc_obj, bind_val_recall_obj,
                               bind_val_precision_obj, bind_val_f1_obj, bind_val_mcc_obj)

            end_time = time.time() - start_time
            print("------------------------//------------------------")
            print((
                          "Binding Vector Epoch: %d, Time Taken: %0.2f, Train Loss: %0.4f, Train Acc: %0.4f, Train Recall: %0.4f, Train Precision: %0.4f, " +
                          "Train F1: %0.4f, Train MCC: %0.4f, Val Loss: %0.4f, Val Acc: %0.4f, Val Recall: %0.4f, Val Precision: %0.4f, " +
                          "Val F1: %0.4f, Val MCC: %0.4f") %
                  (epoch, end_time, float(bind_train_loss_obj.result()), float(bind_train_acc_obj.result()),
                   float(bind_train_recall_obj.result()), float(bind_train_precision_obj.result()),
                   float(bind_train_f1_obj.result()), float(bind_train_mcc_obj.result()),
                   float(bind_val_loss_obj.result()),
                   float(bind_val_acc_obj.result()),
                   float(bind_val_recall_obj.result()),
                   float(bind_val_precision_obj.result()),
                   float(bind_val_f1_obj.result()),
                   float(bind_val_mcc_obj.result())))
            print("------------------------//------------------------")
            ckpt.step.assign_add(1)
            bind_ratio += 1

            if bind_ratio >= bind_vector_epochs:
                training_mode(model, 'affinity')
                affinity_val_loss_obj.reset_states()
                affinity_val_rmse_obj.reset_states()
                affinity_val_ci_obj.reset_states()

                for step, (prot_input, smiles_input, kd_values) in enumerate(affinity_data_val):
                    affinity_test_step(prot_input, smiles_input, kd_values, model, affinity_loss_fn,
                                       affinity_val_loss_obj, affinity_val_rmse_obj, affinity_val_ci_obj)

                affinity_ratio = 0
                bind_ratio = 0
                es_count += 1
                train_cycle += 1

                print("------------------------//------------------------")
                print('---------Training Cycle %d Finished' % train_cycle)
                print("------------------------//------------------------")

                if float(affinity_val_rmse_obj.result()) < best_affinity_metric and \
                        (best_affinity_metric - float(affinity_val_rmse_obj.result())) >= 0.001 and \
                        float(bind_val_mcc_obj.result()) > best_bind_metric and (
                        float(bind_val_mcc_obj.result()) - best_bind_metric) >= 0.001:

                    best_affinity_metric = float(affinity_val_rmse_obj.result())
                    best_bind_metric = float(bind_val_mcc_obj.result())
                    best_epoch = epoch
                    es_count = 0
                    ckpt_manager.save(checkpoint_number=int(best_epoch))

                    logger(("Fold: %d, Train Cycle: %d, Epoch: %d, Bind Vector Train Loss: %0.4f, Bind Vector Train "
                            "Acc: %0.4f, " +
                            "Bind Vector Train Recall: %0.4f, Bind Vector Train Precision: %0.4f, " +
                            "Bind Vector Train F1: %0.4f, Bind Vector Train MCC: %0.4f, Bind Vector Val Loss: %0.4f, " +
                            "Bind Vector Val Acc: %0.4f, Bind Vector Val Recall: %0.4f, Bind Vector Val Precision: "
                            "%0.4f, " +
                            "Bind Vector Val F1: %0.4f, Bind Vector Val MCC: %0.4f, Bind Affinity Train Loss: %0.4f, " +
                            "Bind Affinity Train RMSE: %0.4f, Binding Affinity Train CI: %0.4f, Bind Affinity Val "
                            "Loss: %0.4f, " +
                            "Bind Affinity Val RMSE: %0.4f, Bind Affinity Val CI: %0.4f") % (
                               fold_idx, train_cycle, best_epoch, float(bind_train_loss_obj.result()),
                               float(bind_train_acc_obj.result()),
                               float(bind_train_recall_obj.result()), float(bind_train_precision_obj.result()),
                               float(bind_train_f1_obj.result()), float(bind_train_mcc_obj.result()),
                               float(bind_val_loss_obj.result()), float(bind_val_acc_obj.result()),
                               float(bind_val_recall_obj.result()), float(bind_val_precision_obj.result()),
                               float(bind_val_f1_obj.result()), float(bind_val_mcc_obj.result()),
                               float(affinity_train_loss_obj.result()),
                               float(affinity_train_rmse_obj.result()), float(affinity_train_ci_obj.result()),
                               float(affinity_val_loss_obj.result()), float(affinity_val_rmse_obj.result()),
                               float(affinity_val_ci_obj.result())), fold_idx, FLAGS)
                else:
                    print(('Fold: %d - No Improvement on bind vector validation mcc & bind affinity validation rmse ' +
                           'since epoch: %d') % (fold_idx, best_epoch))

        if es_count == es_value or epoch == epochs - 1:
            return (float(bind_val_loss_obj.result()),
                    float(bind_val_acc_obj.result()),
                    float(bind_val_recall_obj.result()),
                    float(bind_val_precision_obj.result()),
                    float(bind_val_f1_obj.result()),
                    float(bind_val_mcc_obj.result()),
                    float(affinity_val_loss_obj.result()),
                    float(affinity_val_rmse_obj.result()),
                    float(affinity_val_ci_obj.result()))
