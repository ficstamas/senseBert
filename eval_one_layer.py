import tensorflow as tf
from sensebert import SenseBert
from semcor import SemcorReader
import re
import numpy as np
import os
import argparse


def extract_vectors(input_file, output_path, model, gpu, fname, soft_placement, layer):
    reader = SemcorReader()

    device = f'/device:CPU:0'

    print(f"Trying to place on device: {device} with soft placement {soft_placement}")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=soft_placement, log_device_placement=True)) as session:
        with tf.device(device):
            sensebert_model = SenseBert(model, session=session)  # or sensebert-large-uncased

        layer_tensors = []
        reg = "^bert\/encoder\/Reshape_\d*$"

        for n in session.graph_def.node:
            if re.match(reg, n.name):
                layer_tensors.append(n.name)
        print("Evaluating layers...")
        output_path = os.path.join(output_path, model.split('/')[-1])
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        layer_id = 0
        for layer_tensor in [layer_tensors[1:][layer-1]]:
            print(f"Layer {layer}")

            embedding = None
            r = reader.read_sequences(input_file)

            print_limit = 0
            for seq in r:
                input_ids, input_mask = sensebert_model.tokenize([seq])
                layer = session.graph.get_tensor_by_name(f"{layer_tensor}:0")
                vector = layer.eval(feed_dict={sensebert_model.model.input_ids: input_ids,
                                               sensebert_model.model.input_mask: input_mask}, session=session)

                fltr = np.zeros(input_ids[0].__len__(), dtype=np.bool)
                fltr[np.array(input_ids[0]) == 101] = True
                fltr[np.array(input_ids[0]) == 102] = True
                if embedding is None:
                    embedding = vector[0][~fltr, :]
                else:
                    embedding = np.concatenate([embedding, vector[0][~fltr, :]], axis=0)

                if embedding.shape[0] > print_limit:
                    print(f"Tokens done: {embedding.shape[0]}")
                    print_limit += 10000

            print("Saving layer...")
            name = f"{fname}.{model.split('/')[-1]}.layer_{layer}.npy"
            path = os.path.join(output_path, name)
            np.save(path, embedding)
            print("layer saved!")
            layer_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts SenseBert weights by Semcor')
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--in_file',
                        default='/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')
    parser.add_argument('--out_dir', default='/data/ficstamas/representations/')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--soft_placement', action="store_true")
    parser.add_argument('--name', type=str)

    args = parser.parse_args()
    extract_vectors(args.in_file, args.out_dir, args.transformer, args.gpu, args.name, args.soft_placement, args.layer)

