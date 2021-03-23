import tensorflow as tf
import tensorflow_federated as tff
#import tensorflow_federated.simulation.FileCheckpointManager as ckpt
import tensorflow.keras as tk
import collections
import os
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model
from sklearn import metrics
from itertools import zip_longest
from chexpert_parser import load_dataset, feature_description

class LabelAUC(tf.keras.metrics.AUC):
    def __init__(self, label_id, name="label_auc", **kwargs):
        super(LabelAUC, self).__init__(name=name, **kwargs)
        self.label_id = label_id
     
    def update_state(self, y_true, y_pred, **kwargs):
        return super(LabelAUC, self).update_state(y_true[:, self.label_id], y_pred[:, self.label_id], **kwargs)
     
    def result(self):
        return super(LabelAUC, self).result()

class SplitProcess():
	def __init__(self, model_name, model_architecture, input_shape=(224,224,3), output_shape=(14,), checkpoint_folder='./models/'):
		''' Represents a Federated Process - '''
		self.model_name = model_name
		self.model_architecture = model_architecture
		self.number_of_clients = None # Will be populated according to the dataset dictionary
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.checkpoint_folder = None
		self.current_round=1

	def setup(self, dataset_paths, val_dataset_paths, output_folder):
		'''
		Setup the split process from a dictionary containing the client configuration. The top-level keys are the client names, the corresponding keys are dictionaries with a set of parameters: 'path' - the path of the .tfrecord to load, 'take_only': How many samples to take from the dataset (truncation - can be None)
		:param dataset_paths: a dictionary of dictionaries, example {'client_1':{'path': str, 'take_only': None}}
		:param val_dataset_paths: a dictionary of dictionaries containing validation datasets
		:param output_folder: string 
		'''
		self.number_of_clients = len(dataset_paths)
		self.client_list = sorted(dataset_paths.keys())

		print("Creating split dataset for {} clients".format(self.number_of_clients))
		self.train_datasets = {client: load_dataset(dataset_paths[client]['path'], debug=True, take=dataset_paths[client]['take_only'] if 'take_only' in dataset_paths[client] else None) for client in self.client_list}
		self.val_datasets = {client: load_dataset(val_dataset_paths[client]['path'], debug=True, take=val_dataset_paths[client]['take_only'] if 'take_only' in val_dataset_paths[client] else None) for client in self.client_list}

		if not os.path.exists(output_folder):
		    os.makedirs(output_folder)
		    #os.makedirs(output_folder+'/checkpoint')
		    os.makedirs(output_folder+'/tensorboard_log/train')
		    os.makedirs(output_folder+'/tensorboard_log/valid')
		self.path = output_folder
		#self.checkpoint_folder = output_folder+'/checkpoint'
		self.tb_path_train = output_folder+'/tensorboard_log/train'
		self.tb_path_valid = output_folder+'/tensorboard_log/valid'
		self.tb_writer_t = tf.summary.create_file_writer(self.tb_path_train)
		self.tb_writer_v = tf.summary.create_file_writer(self.tb_path_valid)
		self.models_dict = {}
		for client in self.client_list:
			print("Builiding model of {}...".format(client))
			self.models_dict[client] = self.build_model()

		self.metrics = {
			'auc_train' : tf.keras.metrics.AUC(name='auc_train'),
			'auc_train_card' : LabelAUC(label_id=2, name='auc_train_card'),
			'auc_train_edema' : LabelAUC(label_id=5, name='auc_train_edema'),
			'auc_train_cons' : LabelAUC(label_id=6, name='auc_train_cons'),
			'auc_train_atel' : LabelAUC(label_id=8, name='auc_train_atel'),
			'auc_train_peff' : LabelAUC(label_id=10, name='auc_train_peff'),

			'auc_valid' : tf.keras.metrics.AUC(name='auc_valid'),
			'auc_valid_card' : LabelAUC(label_id=2, name='auc_valid_card'),
			'auc_valid_edema' : LabelAUC(label_id=5, name='auc_valid_edema'),
			'auc_valid_cons' : LabelAUC(label_id=6, name='auc_valid_cons'),
			'auc_valid_atel' : LabelAUC(label_id=8, name='auc_valid_atel'),
			'auc_valid_peff' : LabelAUC(label_id=10, name='auc_valid_peff'),
		}

		self.metrics_name = [met for met in self.metrics.keys()]
		self.loss_fn = tf.keras.losses.BinaryCrossentropy()
		self.optimizer = tf.keras.optimizers.SGD(1e-3)


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


	def compute_metrics(self, y_true, y_pred, client, run):
		if run == 'training':
			self.metrics['auc_train'].update_state(y_true, y_pred)
			self.metrics['auc_train_card'].update_state(y_true, y_pred)
			self.metrics['auc_train_edema'].update_state(y_true, y_pred)
			self.metrics['auc_train_cons'].update_state(y_true, y_pred)
			self.metrics['auc_train_atel'].update_state(y_true, y_pred)
			self.metrics['auc_train_peff'].update_state(y_true, y_pred)
		if run == 'validation':
			self.metrics['auc_valid'].update_state(y_true, y_pred)
			self.metrics['auc_valid_card'].update_state(y_true, y_pred)
			self.metrics['auc_valid_edema'].update_state(y_true, y_pred)
			self.metrics['auc_valid_cons'].update_state(y_true, y_pred)
			self.metrics['auc_valid_atel'].update_state(y_true, y_pred)
			self.metrics['auc_valid_peff'].update_state(y_true, y_pred)


	@tf.function
	def train_step(self, x, y, client):
		with tf.GradientTape(persistent=True) as tape:
			output = self.models_dict[client](x, training=True)												
			loss_value = self.loss_fn(y, output)
		grads = tape.gradient(loss_value, self.models_dict[client].trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.models_dict[client].trainable_weights))
		del tape
		self.compute_metrics(y, output, client, run='training')
		return loss_value

	@tf.function
	def validation_step(self, x, y, client):
		output = self.models_dict[client](x, training=False)
		loss = self.loss_fn(y, output)
		self.compute_metrics(y, output, client, run='validation')
		return loss

	def save_server_weights(self, client, sl_bottom, sl_top=None):
		if sl_top is None:
			weights = []
			for i in range(sl_bottom, len(self.models_dict[client].layers)):
				weights.append(self.models_dict[client].layers[i].get_weights())
		else:
			print("NOT IMPLEMENTED") 
		return weights

	def update_server_weights(self, weights, sl, client_updated="Dummy"):
		for client, model in self.models_dict.items():
			if client is not client_updated:
				for i in range(len(weights)):
					self.models_dict[client].layers[i + sl].set_weights(weights[i])

	def average_weights(self, sl_bottom):
		weights = []
		layers = tuple([ l for l in m.layers[60:] if l.trainable] for m in self.models_dict.values())
		for clients_layers in zip(*layers):
			mean_value = np.mean([l.get_weights() for l in clients_layers], axis=0)
			weights.append(mean_value)
		return weights

	def log_epoch(self, log, epoch, run):
		stacked = np.stack([met.result().numpy() for met in self.metrics.values()], axis=0)
		step_log = pd.DataFrame(np.array([stacked]), columns=self.metrics_name)
		step_log.insert(0, 'Epoch', epoch)
		log = log.append(step_log, ignore_index=True)
		return log

	def save_log(self, log):
		file=self.path+'/log.csv'
		with open(file, mode='w') as f:
			log.to_csv(f, index=False)

	def iterative_training(self, split_layer, epochs):
		logger = pd.DataFrame()
		for e in range(epochs):
			print("Start of epoch %d" %(e)) 
			step=0
			# Training
			for row0, row1, row2 in zip_longest(self.train_datasets['client_0'], self.train_datasets['client_1'], self.train_datasets['client_2']):
				step+=1
				if row0:
					train_loss = self.train_step(row0['x'], row0['y'], 'client_0')
					server_weights = self.save_server_weights('client_0', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_0')

				if row1:
					train_loss = self.train_step(row1['x'], row1['y'], 'client_1')
					server_weights = self.save_server_weights('client_1', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_1')           

				if row2:
					train_loss = self.train_step(row2['x'], row2['y'], 'client_2')
					server_weights = self.save_server_weights('client_2', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_2')
				if step%50==0:
					template = 'TRAINING: Epoch {}, Step {}, AUC train: {}, AUC train cardiomegaly: {}'
					print(template.format(e, step, self.metrics['auc_train'].result().numpy(), self.metrics['auc_train_card'].result().numpy()))
			
			# Saving training metrics on tensorboard
			with self.tb_writer_t.as_default():
				for met in dict(list(self.metrics.items())[:6]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Validation
			for client in self.client_list:
				for step, row in enumerate(self.val_datasets[client]):
					val_loss = self.validation_step(row['x'], row['y'], client)
					if step%50==0:
						template = 'VALIDATION: Epoch {}, Step {}, AUC valid: {}, Auc valid card: {}'
						print(template.format(e, step, self.metrics['auc_valid'].result().numpy(), self.metrics['auc_valid_card'].result().numpy()))

			# Saving validation metrics on tensorboard
			with self.tb_writer_v.as_default():
				for met in dict(list(self.metrics.items())[6:]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			#ckpt.save(outputFolder+"/Model_A_epoch_{}".format(e))

			# Log metrics 
			logger = self.log_epoch(logger, e, run='training')
			print(logger)


			# Reset metrics at each epoch
			for m in self.metrics.values():
				m.reset_states()

		# Save Models and Logs 
		self.models_dict['client_0'].save(self.path+'/client_0.h5')
		self.models_dict['client_1'].save(self.path+'/client_1.h5')
		self.models_dict['client_0'].save(self.path+'/client_2.h5')
		self.save_log(logger)

	def evaluation(self, epochs, model_path):
		logger = pd.DataFrame()
		print("Carico model0 from {}".format(model_path+'/client_0.h5'))
		model0 = load_model(model_path+'/client_0.h5')
		print("Carico model1 from {}".format(model_path+'/client_1.h5'))
		model1 = load_model(model_path+'/client_1.h5')
		print("Carico model2 from {}".format(model_path+'/client_2.h5'))
		model2 = load_model(model_path+'/client_2.h5')

		self.models_dict['client_0'] = model0
		self.models_dict['client_1'] = model1
		self.models_dict['client_2'] = model2

		for e in range(epochs):
			print("Start of epoch %d" %(e))
			step=0
			for client in self.client_list:
				for step, row in enumerate(self.val_datasets[client]):
					val_loss = self.validation_step(row['x'], row['y'], client)
					if step%50==0:
						template = 'VALIDATION: Epoch {}, Step {}, AUC valid: {}, AUC_cardiomegaly: {}, AUC_edema: {}, AUC_consolidation: {}, AUC_atelectasis: {}, AUC_pleural_effusion: {}'
						print(template.format(e, step, self.metrics['auc_valid'].result().numpy(), self.metrics['auc_valid_card'].result().numpy(), self.metrics['auc_valid_edema'].result().numpy(), self.metrics['auc_valid_cons'].result().numpy(), self.metrics['auc_valid_atel'].result().numpy(), self.metrics['auc_valid_peff'].result().numpy()))

			# Log metrics 
			logger = self.log_epoch(logger, e, run='training')
			print(logger)

			# Reset metrics at each epoch
			for m in self.metrics.values():
				m.reset_states()

		self.save_log(logger)


	def parallel_training(self, split_layer, epochs):
		logger = pd.DataFrame()
		for e in range(epochs):
			print("Start of epoch %d" %(e)) 
			step=0
			# Training
			for row0, row1, row2 in zip_longest(self.train_datasets['client_0'], self.train_datasets['client_1'], self.train_datasets['client_2']):
				step+=1
				if row0:
					train_loss = self.train_step(row0['x'], row0['y'], 'client_0')
				if row1:
					train_loss = self.train_step(row1['x'], row1['y'], 'client_1')
				if row2:
					train_loss = self.train_step(row2['x'], row2['y'], 'client_2')
				server_weights = self.average_weights(split_layer)
				self.update_server_weights(server_weights, split_layer)
				if step%50==0:
					template = 'TRAINING: Epoch {}, Step {}, AUC train: {}, AUC train cardiomegaly: {}'
					print(template.format(e, step, self.metrics['auc_train'].result().numpy(), self.metrics['auc_train_card'].result().numpy()))

			# Saving training metrics on tensorboard
			with self.tb_writer_t.as_default():
				for met in dict(list(self.metrics.items())[:6]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Validation
			for client in self.client_list:
				for step, row in enumerate(self.val_datasets[client]):
					val_loss = self.validation_step(row['x'], row['y'], client)
					if step%50==0:
						template = 'VALIDATION: Epoch {}, Step {}, AUC valid: {}, Auc valid card: {}'
						print(template.format(e, step, self.metrics['auc_valid'].result().numpy(), self.metrics['auc_valid_card'].result().numpy()))				

			# Saving validation metrics on tensorboard
			with self.tb_writer_v.as_default():
				for met in dict(list(self.metrics.items())[6:]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Log metrics 
			logger = self.log_epoch(logger, e, run='training')
			print(logger)

			# Reset metrics at each epoch
			for m in self.metrics.values():
				m.reset_states()

		# Save Models and Logs 
		self.models_dict['client_0'].save(self.path+'/client_0.h5')
		self.models_dict['client_1'].save(self.path+'/client_1.h5')
		self.models_dict['client_0'].save(self.path+'/client_2.h5')
		self.save_log(logger)

