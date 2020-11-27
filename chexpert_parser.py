import tensorflow as tf
import collections

# Parsing TFRecord


feature_description = {
    'age':                        tf.io.FixedLenFeature([], tf.float32),
    'frontal/lateral':            tf.io.FixedLenFeature([], tf.float32),
    'ap/pa':                      tf.io.FixedLenFeature([], tf.float32),
    'no finding':                 tf.io.FixedLenFeature([], tf.float32),
    'enlarged cardiomediastinum': tf.io.FixedLenFeature([], tf.float32),
    'cardiomegaly':               tf.io.FixedLenFeature([], tf.float32),
    'lung opacity':               tf.io.FixedLenFeature([], tf.float32),
    'lung lesion':                tf.io.FixedLenFeature([], tf.float32),
    'edema':                      tf.io.FixedLenFeature([], tf.float32),
    'consolidation':              tf.io.FixedLenFeature([], tf.float32),
    'pneumonia':                  tf.io.FixedLenFeature([], tf.float32),
    'atelectasis':                tf.io.FixedLenFeature([], tf.float32),
    'pneumothorax':               tf.io.FixedLenFeature([], tf.float32),
    'pleural effusion':           tf.io.FixedLenFeature([], tf.float32),
    'pleural other':              tf.io.FixedLenFeature([], tf.float32),
    'fracture':                   tf.io.FixedLenFeature([], tf.float32),
    'support devices':            tf.io.FixedLenFeature([], tf.float32),
    'patient_id':                 tf.io.FixedLenFeature([], tf.float32),
    'image':                      tf.io.FixedLenFeature([], tf.string),
}

def parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above and crop the image.
  example_parsed = tf.io.parse_example(example_proto, feature_description)
  
  cropped_image = tf.image.resize_with_crop_or_pad(tf.io.decode_png(example_parsed["image"],channels=3), 224 , 224)
  example_parsed["image"] = tf.image.convert_image_dtype(cropped_image, tf.float32)
  for k in example_parsed.keys():
    if k != "image":
      if tf.math.is_nan(example_parsed[k]):
        example_parsed[k] = tf.constant(0.)
      if tf.math.less(example_parsed[k], 0.):
        tf.random.uniform([1], minval=0.55, maxval=0.85)

  return example_parsed






def load_dataset(path, BATCH_SIZE = 32, SHUFFLE_BUFFER = 128, PREFETCH_BUFFER=1, debug=False, take=None):
    if debug:
        print("loading {}".format(path))
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.take(take) if take is not None else dataset
    def batch_format_fn(element):
        # Flatten a batch `pixels` and return the features as an `OrderedDict`.
        return collections.OrderedDict( x = element['image'],
                                        y =  [ element['no finding'], element['enlarged cardiomediastinum'], element['cardiomegaly'], element['lung opacity'], element['lung lesion'],
                                              element['edema'], element['consolidation'], element['pneumonia'], element['atelectasis'], element['pneumothorax'], element['pleural effusion'],
                                              element['pleural other'], element['fracture'], element['support devices'] ]
                                       )
    parsed_dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #return parsed_dataset.shuffle(SHUFFLE_BUFFER).repeat(NUM_EPOCHS).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
    return parsed_dataset.map(batch_format_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)