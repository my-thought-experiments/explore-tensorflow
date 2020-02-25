# TensorFlow Hub

* TensorFlow Hub is a library for reusable ML modules. It used for the `publication`, `discovery`, and `consumption` of reusable parts of ML models.
* A module is a self-contained piece of a `TensorFlow graph`, along with its `weights` and `assets`, that can be reused across different tasks in a process known as `transfer learning`.

Transfer learning can:

* Train a model with a smaller dataset.
* Improve generalization.
* Speed up training.

## Pre-requisite understanding

### 1. Module instantiation

1. Modules made up with different models ([Inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202), ResNet, ElMo etc) serving different purposes (image classification, text embeddings etc) are hosted in TensorFlow Hub website.
2. The user has to browse through the catalogue of modules and then once finalised with his purpose and model, needs to copy the URL of the model where it is hosted.
3. Then, the user can instantiate his module as below:

```
import tensorflow_hub as hub
module = hub.Module(<<Module URL as string>>, trainable=True)
```

Notes:

* Apart from the URL parameter, the other most notable parameter is `trainable`. If user wishes to fine-tune/modify the weights of the model, this parameter has to be set as True.

### 2. Signature

* The signature of the module specifies what is the purpose for which module is being used for.
* All the module, comes with the ‘default’ signature, if a signature is not explicitly mentioned.
* When ‘default’ signature is used, the internal layers of the model is abstracted from the user.
* `get_signature_names()` function can be used to list out all the signature names of the module.

```
import tensorflow_hub as hub

module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
print(module.get_signature_names())
```

### 3. Expected inputs

* Each of the module has some set of expected inputs depending upon the signature of the module being used.
* Most of the modules have documented the set of expected inputs in TensorFlow Hub website (particularly for ‘default’ signature), some of them haven’t.
* `get_input_info_dict()` function is used to obtain the expected input along with it’s size and datatype.

```
import tensorflow_hub as hub

module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
print(module.get_input_info_dict())   # When no signature is given, considers it as 'default'

# {'images': <hub.ParsedTensorInfo shape=(?, 299, 299, 3) dtype=float32 is_sparse=False>}

print(module.get_input_info_dict(signature='image_feature_vector'))

# {'images': <hub.ParsedTensorInfo shape=(?, 299, 299, 3) dtype=float32 is_sparse=False>}
```

### 4. Expected outputs

* In order to build the remaining part of the graph after the TensorFlow Hub’s model is built, it is necessary to know the expected type of output. `get_output_info_dict()` function is used for this purpose. 

Notes:

* For `default` signature, there will be usually only 1 output, but when you use a non-default signature, multiple layers of the graph will be exposed to you.

```
import tensorflow_hub as hub

module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
print(module.get_output_info_dict())  # When no signature is given, considers it as 'default'

# {'default': <hub.ParsedTensorInfo shape=(?, 1001) dtype=float32 is_sparse=False>}

print(module.get_output_info_dict(signature='image_classification'))

# {'InceptionV3/global_pool': <hub.ParsedTensorInfo shape=(?, 1, 1, 2048) dtype=float32 is_sparse=False>,
# 'InceptionV3/Logits': <hub.ParsedTensorInfo shape=(?, 1001) dtype=float32 is_sparse=False>,
# 'InceptionV3/Conv2d_2b_3x3': <hub.ParsedTensorInfo shape=(?, 147, 147, 64) dtype=float32 is_sparse=False>
# ..... Several other exposed layers...... }
```

### 5. Collecting required layers of module

* After instantiating the module, you have to extract your desired layer/output from module and add it to the graph.

```
import tensorflow as tf
import tensorflow_hub as hub

images = tf.placeholder(tf.float32, (None, 299, 299, 3))

module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
logits1 = module(dict(images=images))    # implies default signature
print(logits1)
# Tensor("module_apply_default/InceptionV3/Logits/SpatialSqueeze:0", shape=(?, 1001), dtype=float32)

module_features = module(dict(images=images), signature='image_classification', as_dict=True)
# module_features stores all layers in key-value pairs

logits2 = module_features['InceptionV3/Logits']
print(logits2)
# Tensor("module_apply_image_classification/InceptionV3/Logits/SpatialSqueeze:0", shape=(?, 1001), dtype=float32)

global_pool = module_features['InceptionV3/global_pool']
print(global_pool)
# Tensor("module_apply_image_classification/InceptionV3/Logits/GlobalPool:0", shape=(?, 1, 1, 2048), dtype=float32)
```

### 6. Initialising TensorFlow Hub operations

* The resulting output weight/values of all operations present in modules are hosted by TensorFlow Hub in a tabular format.
* This needs to be initialized using `tf.tables_initializer()` along with the initializations of regular variables.

```
import tensorflow as tf

with tf.Session() as sess:
  sess.run([tf.tables_initializer(), <<other initializers>>])
```

## Code skeleton block

Once the complete graph comprising of your module, learning algorithm optimisers, objective function, custom layers etc is built, this is how the graph part of your code may look like.

```
import tensorflow as tf
import tensorflow_hub as hub

<< Create Placeholders >>
<< Create Dataset and Iterators >>

module1 = hub.Module(<< Module URL >>)
logits1 = module1(<< input_dict >>)

module2 = hub.Module(<< Module URL >>)
module2_features = module2(<< input_dict >>, signature='default', as_dict=True)
logits2 = module2_features['default']

<< Remaining graph, learning algorithms, objective function, etc >>

with tf.Session() as sess:
  sess.run([tf.tables_initializer(), << other initializers >>])

  << Remaining training pipeline >>
```

* The first module is built using bare minimum code which implicitly uses default signature and layer.
* In the second module, I am explicitly specifying default signature and layer.

In the similar manner, we can specify the non-default signature and layers.



## References

* [TensorFlow Hub Official Website](https://www.tensorflow.org/hub/)
* [TensorFlow Hub Official Hosting](https://tfhub.dev/)
* [TensorFlow Hub Self Hosting](https://www.tensorflow.org/hub/hosting)
* [Intro Blog](https://blog.tensorflow.org/2018/03/introducing-tensorflow-hub-library.html)
