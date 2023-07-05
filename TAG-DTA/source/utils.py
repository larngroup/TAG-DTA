from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
import tensorflow as tf
import numpy as np
from main import *
import matplotlib.pyplot as plt
import seaborn as sns
from layers_utils import *

# Root Mean Squared Error Function
def rmse_function(y_true, y_pred):
    rmse = tf.math.sqrt(tf.math.reduce_mean((tf.expand_dims(y_true, axis=-1) - y_pred) ** 2))
    return rmse

# Concordance Index Function
def c_index(y_true, y_pred):
    """
    Concordance Index Function
    Args:
    - y_trues: true values
    - y_pred: predicted values
    """

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

# Affinity Evaluation Metrics
def inference_metrics(real,preds):
    """
    Prediction Efficiency Evaluation Metrics
    Args:
    - real: true values
    - preds: predicted values
    """


    return mse(real, preds), mse(real, preds, squared=False),\
        c_index(real, preds).numpy(), r2s(real, preds),\
        stats.spearmanr(real, preds)[0]


# Load Saved TAG-DTA Model
def load_saved_model(model,
                     smiles_opt = ['radam', '1e-05', '0.9', '0.999', '1e-08', '1e-05', '0', '0', '0'] ,
                     bind_opt = ['radam', '1e-04', '0.9', '0.999', '1e-08', '1e-05', '0', '0', '0'],
                     aff_opt = ['radam', '1e-04', '0.9', '0.999', '1e-08', '1e-05', '0', '0', '0'],
                     init_bias = [-1.663877], ckpt_path='../checkpoint/tag_dta_checkpoint/'):

    smiles_opt = opt_config(smiles_opt)
    bind_opt = opt_config(bind_opt)
    aff_opt = opt_config(aff_opt)
    init_bias = np.array(init_bias)
    model.get_layer('binding_output_block').get_layer('mlp_out').set_weights(
        [model.get_layer('binding_output_block').get_layer('mlp_out').get_weights()[0]] + [init_bias])
    global_var = tf.Variable(1)
    ckpt_obj = tf.train.Checkpoint(step=global_var, smiles_bert_opt=smiles_opt, bind_opt=bind_opt,
                                   affinity_opt=aff_opt, model=model)
    latest = tf.train.latest_checkpoint(ckpt_path)
    ckpt_obj.restore(latest).expect_partial()
    model = ckpt_obj.model

    return model

# Binding Affinity Predictions vs True Values Scatter Plot
def pred_scatter_plot(real_values, pred_values, title, xlabel, ylabel, savefig, figure_name):
    fig, ax = plt.subplots()
    ax.scatter(real_values, pred_values, c='Red',
               edgecolors='black')
    # ax.plot([real_values.min(),real_values.max()],[real_values.min(),real_values.max()],'k--',lw = 4)
    ax.plot(real_values, real_values, 'k--', lw=4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.title(title)
    plt.xlim([4, 11])
    plt.ylim([4, 11])
    plt.show()
    if savefig:
        fig.savefig(figure_name, dpi=1200, format='eps')