import tensorflow as tf
import tensorflow_federated as tff
import tensorflow.keras as tk
import collections
from chexpert_parser import load_dataset, feature_description

class FederatedProcess():
    def __init__(self, model_name, model_architecture, input_shape=(224,224,3), output_shape=(14,), checkpoint_folder='./models/'):
        ''' Represents a Federated Process - '''
        self.model_name = model_name
        self.model_architecture = model_architecture
        self.number_of_clients = None # Will be populated according to the dataset dictionary
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.checkpoint_folder = checkpoint_folder
        self.current_round=1

    def setup(self, dataset_paths, batch_size):
        '''
        Setup the federated process from a dictionary containing the client configuration. The top-level keys are the client names, the corresponding keys are dictionaries with a set of parameters: 'path' - the path of the .tfrecord to load, 'take_only': How many samples to take from the dataset (truncation - can be None)
        :param dataset_paths: a dictionary of dictionaries, example {'client_1':{'path': str, 'take_only': None}}
        '''
        # Aggiungere come parametri tutto ciò che si vuole parametrizzare durante gli esperimenti
        self.number_of_clients = len(dataset_paths)
        self.client_list = sorted(dataset_paths.keys())
        print("Creating federated dataset for {} clients".format(self.number_of_clients))
        client_datasets = {client: load_dataset(dataset_paths[client]['path'], debug=True, take=dataset_paths[client]['take_only'] if 'take_only' in dataset_paths[client] else None) for client in self.client_list}
        self.client_data = tff.simulation.ClientData.from_clients_and_fn(self.client_list, lambda client: client_datasets[client])
        self.federated_train_data = [self.client_data.create_tf_dataset_for_client(client) for client in self.client_list]
        # TODO: Eventualmente implementare il sampling dei client
        # È necessaria questa riga?
        # sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(self.federated_train_data[0])))

        print("Defining federated averaging process...")
        self.iterative_process = tff.learning.build_federated_averaging_process(
            self.federated_model_function,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.8))
        self.signature = self.iterative_process.initialize.type_signature
        print("Initializing federated process...")
        self.state = self.iterative_process.initialize()
        print("Ready.")

        #TODO: Implementare save/load dei checkpoint per il federated

    def build_model(self):
        '''
        Builds a keras model based on the model architecture provided in the init function
        :return: A Keras model
        '''
        base_model = self.model_architecture(input_shape=self.input_shape, weights='imagenet', include_top=False)
        x = base_model.output
        x = tk.layers.GlobalAveragePooling2D()(x)
        predictions = tk.layers.Dense(14, activation='sigmoid')(x)
        return tk.Model(inputs=base_model.inputs, outputs=predictions)

    def federated_model_function(self):
        '''
        Builds the federated model. This function shouldn't be called directly but rather passed as parameter to tensorflow_federated
        :return: a tff.learning model
        '''
        self.input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=[None] + [d for d in self.input_shape], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None] + [d for d in self.output_shape], dtype=tf.float32)
        )
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = self.build_model()

        return tff.learning.from_keras_model(
            keras_model,
            input_spec=self.input_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryCrossentropy()])

    def log_federated_round(self, current_round, metrics):
        '''
        Called when a federated round ends. Log information about the current state on tensorboard and on a separate csv.
        :return:
        '''
        # TODO: Implement this
        pass

    def federated_round(self, rounds):
        for r in range(self.current_round, self.current_round+rounds):
            self.state, metrics = self.iterative_process.next(self.state, self.federated_train_data)
            print('round {:2d}, metrics={}'.format(r, metrics))
            self.log_federated_round(r, metrics)
            self.current_round+=1

