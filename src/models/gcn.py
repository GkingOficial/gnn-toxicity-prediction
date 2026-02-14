import numpy as np
# import os
# import csv
# import random
# import itertools
from utils.utils import load_folds

# from utils.utils import shuffle_two_list, split_train_eval_test, load_input_HIV
# from rdkit import Chem

from mc_dropout import mc_dropout
import tensorflow as tf
# from rdkit.Chem.AllChem import Compute2DCoords
# from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

# from rdkit.Chem.Draw import IPythonConsole
# from IPython.display import SVG

# from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
# from IPython.display import SVG
# from rdkit import rdBase

np.set_printoptions(precision=3)

from training.train_gcn import training

def build_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    dim1 = 62
    dim2 = 768
    max_atoms = 170
    num_layer = 4
    batch_size = 8
    epoch_size = 200
    learning_rate = 0.0003
    regularization_scale = 4e-4
    beta1 = 0.9
    beta2 = 0.98

    flags.DEFINE_string('task_type', 'classification', '')
    flags.DEFINE_integer('hidden_dim', dim1, '')
    flags.DEFINE_integer('latent_dim', dim2, '')
    flags.DEFINE_integer('max_atoms', max_atoms, '')
    flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
    flags.DEFINE_integer('num_attn', 4, '# of heads for multi-head attention')
    flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
    flags.DEFINE_integer('epoch_size', epoch_size, 'Epoch size')
    # flags.DEFINE_integer('num_train', num_train, 'Number of training data')
    flags.DEFINE_float('regularization_scale', regularization_scale, '')
    flags.DEFINE_float('beta1', beta1, '')
    flags.DEFINE_float('beta2', beta2, '')
    flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp')
    flags.DEFINE_float('init_lr', learning_rate, 'Batch size')
    return FLAGS

def main():
    
    # smi_total, prop_total = load_input_HIV()
    # num_total = len(smi_total)
    # num_test = int(num_total * 0.2)
    # num_train = num_total - num_test
    # num_eval = int(num_train * 0.1)
    # num_train -= num_eval

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Set FLAGS for environment setting
    FLAGS = build_flags()

    model_name = 'MC-Dropout_HIV'
    print("Do Single-Task Learning")
    print("Hidden dimension of graph convolution layers:", FLAGS.hidden_dim)
    print("Hidden dimension of readout & MLP layers:", FLAGS.latent_dim)
    print("Maximum number of allowed atoms:", FLAGS.max_atoms)
    print("Batch sise:", FLAGS.batch_size, "Epoch size:", FLAGS.epoch_size)
    print("Initial learning rate:", FLAGS.init_lr, "\t Beta1:", FLAGS.beta1, "\t Beta2:", FLAGS.beta2,
        "for the Adam optimizer used in this training")

    # model = mc_dropout(FLAGS)
    # training(model, FLAGS, model_name, smi_total, prop_total)

    model_name = "Mutag200GCN"
    folds = load_folds()
    for i in range(5):
        print(f"\n===== Fold {i+1}/5 =====")
        
        smi_train = []
        prop_train = []
        
        for j, (smi, prop) in enumerate(folds):
            if j == i:
                smi_test = smi
                prop_test = prop
            else:
                smi_train.extend(smi)
                prop_train.extend(prop)
        
        smi_train = np.array(smi_train)
        prop_train = np.array(prop_train)
        smi_test = np.array(smi_test)
        prop_test = np.array(prop_test)

        unique, counts = np.unique(prop_train, return_counts=True)
        print("Treino:", dict(zip(unique, counts)))

        unique, counts = np.unique(prop_test, return_counts=True)
        print("Teste:", dict(zip(unique, counts)))

        tf.compat.v1.reset_default_graph()
        model = mc_dropout(FLAGS)
        
        training(model, FLAGS, f"{model_name}_fold{i}", smi_train, prop_train, smi_test, prop_test, i)
        # input('CONTINUAR ==>')

if __name__ == "__main__":
    main()