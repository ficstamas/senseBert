import tensorflow as tf
from sensebert import SenseBert
from semcor import SemcorReader
import re
import numpy as np
import os
import argparse


def extract_vectors(input_file, output_path, model):
    reader = SemcorReader()

    with tf.Session() as session:
        sensebert_model = SenseBert(model, session=session)  # or sensebert-large-uncased
        # summ_writer = tf.summary.FileWriter(os.path.join('summaries', 'first'), session.graph)
        layer_tensors = []
        reg = "^bert\/encoder\/Reshape_\d*$"
        for n in tf.get_default_graph().as_graph_def().node:
            if re.match(reg, n.name):
                layer_tensors.append(n.name)
        for layer_id, layer_tensor in enumerate(layer_tensors[1:]):
            embedding = None
            r = reader.read_sequences(input_file)
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
            name = f"{input_file.split('.')[0]}.{model}.layer_{layer_id}.npy"
            path = os.path.join(output_path, name)
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            np.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts SenseBert weights by Semcor')
    parser.add_argument('--transformer', required=True,
                        choices=['sensebert-base-uncased', 'sensebert-large-uncased'])
    parser.add_argument('--in_file', default='/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')
    parser.add_argument('--out_dir', default='/data/ficstamas/representations/')
