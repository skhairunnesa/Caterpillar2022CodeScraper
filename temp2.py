from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from contracts import contract, new_contract
from keras import backend as K
import numpy as np
from contract_checker_library import ContractException  # Assuming you have a ContractException class in your library
## For Post 3
# Define a contract checking function for build_fn
@new_contract
def keras_model_contract(model):
    if not isinstance(model, Sequential):
        raise ContractException("The build_fn should return a Keras Sequential model.")

# Your nn_model function
def nn_model():
    return
    # ... your model creation logic here ...

# Contract checking function
@contract(build_fn='callable,keras_model_contract')
def create_nn_model():
    return nn_model()

# Wrap the neural network model function as a regressor using KerasRegressor
regressor = KerasRegressor(build_fn=create_nn_model, nb_epoch=2)
##For Post 12
# Define a contract checking function for model usage
@new_contract
def keras_model_usage_contract(model):
    if not isinstance(model, Sequential):
        raise ContractException("The model should be an instance of Keras Sequential.")
    if not model.compiled:
        raise ContractException("The model has not been compiled. Please compile the model before training.")
# Load weights and make predictions contract checking function
@contract(model='callable,keras_model_usage_contract', weights='str', input_data='array_like')
def load_weights_and_predict(model, weights, input_data):
    model.load_weights(weights)
    return model.predict(input_data)

# Excel sheet row 6
@new_contract
def reference_to_model_input_contract(ref):
    if ref != model.input:
        raise ContractException("Reference should be made to the model's input.")
    
# excel sheet row 7
@new_contract
def contract_checker1(model):
    concatenate_layers = [layer for layer in model.layers if isinstance(layer, Concatenate)]

    for concatenate_layer in concatenate_layers:
        if concatenate_layer.input_shape is None or None in concatenate_layer.input_shape:
            raise ContractException("Input shape not specified for a Concatenate layer.")

#excel sheet row 10
@new_contract
def batch_norm_order(model):
    for i in range(1, len(model.layers) - 1):  # Iterate from the second layer to the second-to-last layer
     current_layer = model.layers[i]
     previous_layer = model.layers[i - 1]
     next_layer = model.layers[i + 1]

     if isinstance(current_layer, BatchNormalization):
            if isinstance(previous_layer, Dense) and not isinstance(next_layer, Dense):
                break
            else:
                    raise ContractException("Invalid layer configuration: The layer before Batch Normalization should be Dense(linear-layer), and the layer after should be non-linear.")
            

#excel sheet row 12
@new_contract
def contract_checker_PReLU(model):
    for layer in model.layers:
        if isinstance(layer, PReLU):
        # Check if PReLU layer is wrapped with an activation layer
            if len(layer._layers) > 0 and isinstance(layer._layers[0], Activation):
                raise ContractException("PReLU layer is wrapped with an Activation layer.")

#excel sheet row 16
@new_contract           
def contract_check_sequential_model(model,input_data, target_data):
    # Check if the model's output dimensions match target data dimensions
    msg1=""
    msg2=""
    model_output = model.predict(input_data)
    if model_output.shape[1] != target_data.shape[1]:
        # Check if the LSTM layer has return sequence set to true
        lstm_layers = [layer for layer in model.layers if isinstance(layer, LSTM)]

        for lstm_layer in lstm_layers:
            if not lstm_layer.return_sequences:
                msg1+="LSTM layer {lstm_layer.name} does not have return_sequences set to True."
                
        # Check if the Dense layer is wrapped in TimeDistributed
        dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

        for dense_layer in dense_layers:
             if not any(isinstance(wrapper, TimeDistributed) for wrapper in dense_layer._layers):
                msg2+="Dense layer {dense_layer.name} is not wrapped in TimeDistributed."
    raise ContractException(msg1+msg2)
 
#this week
 #excel sheet row 17
@new_contract
def contract_check_concat_parameters(input_b, intermediate_from_a):
    # Get the output shapes of the two layers
    if isinstance(intermediate_from_a, np.ndarray):
        raise ContractException("intermediate_from_a is a NumPy array. Do not use predict()")
    else:
        shape1 = K.int_shape(input_b)
        shape2 = K.int_shape(intermediate_from_a)

        # Check if the dimensions are compatible for concatenation
        if shape1 and shape2 and shape1[1] == shape2[1]:
            return True
        else:
            raise ContractException("The dimensions of the two layers are not compatible for concatenation.")

#excel sheet row 19 
@new_contract
def contract_cnn_with_lstm(model):
    # Check if the model has a convolutional layer followed by an LSTM layer
    for i in range(0, len(model.layers) - 1):  # Iterate from the first layer to the second-to-last layer
     current_layer = model.layers[i]
     next_layer = model.layers[i + 1]
     if isinstance(current_layer, Conv2D):
            if isinstance(next_layer, LSTM):
                cnn_output_shape = current_layer.output_shape
                lstm_input_shape = next_layer.input_shape
                # Check if the output shape of the CNN and input shape of the LSTM are compatible
                if cnn_output_shape[-1] != lstm_input_shape[-1]:
                    raise ContractException("Output shape of Conv2D do not match input shape of LSTM, Use 'TimeDistributed' to wrapper on the CNN layer.")

#excel sheet row 22
@new_contract
def check_reset_weights(model):
    initial_weights = tf.keras.models.load_model('initial_weights.h5').get_weights()
    #This loop iterates over the layers of the model and the corresponding initial weights loaded from the saved file.
    for layer, initial_weight in zip(model.layers, initial_weights):
            current_weight = layer.get_weights()
            #This line checks whether all elements of the current_weight tensor are equal to the initial_weight tensor using TensorFlow operations.
            if not tf.reduce_all(tf.equal(current_weight, initial_weight)):
                raise ContractException("Weights are not reset to initial weights.")

#excel sheet row 25
@new_contract
def check_BN_updateOps(model, X_train, y_train):
    # Start a TensorFlow session
    with tf.compat.v1.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Get the update operations from model
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Run the update operations and the forward pass
        sess.run([update_ops, model.output], feed_dict={model.input: X_train, model.output: y_train})

        # Check if update operations were executed
        if not update_ops:
            raise ContractException("Batch Normalization statistics are not updated during training. "
                                    "You need to manually add the update operations.")    
    
#excel sheet row 65
@new_contract
def check_mergeLayer_input(model):
    for layer in model.layers:
        if isinstance(layer, Merge):
            # Access the inputs property of the merge layer
            input_types = layer.inputs

            # Check if the input types are instances of Keras models
            for input_type in input_types:
                if  isinstance(input_type, Model):
                    raise ContractException("Use functional API merge layers Add() or substract()"
                                             " to merge output of two models")
                
#excel sheet row 72
@new_contract
def check_multi_initialization(model):
    if K.backend() == 'tensorflow':
        raise ContractException("The Backend used is tensorflow, please use clear_session()"
                                " after usage of model in loop ")
    
#excel sheet row 81
@new_contract
def check_conv1d_input_shape(model):
    for layer in model.layers:
        if isinstance(layer, Conv1D):
            if layer.input_shape is not None and (len(layer.input_shape) != 3):
                raise ContractException("The layer does not have a spatial dimension in its input shape. "
                                "Expected input shape (batch_size, steps, features).")
            
#excel sheet row 84
@new_contract
def check_conv2d_input_shape(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            if layer.input_shape is not None and (len(layer.input_shape) != 3):
                raise ContractException("input_shape should contain 3 dimensions only,"
                                        " Expected input shape (height, width, channels).")

#excel sheet row 89
@new_contract
def check_embedding_argument(model,data):
    # Create a Tokenizer
    tokenizer = Tokenizer()
    # Fit the tokenizer on the text data
    tokenizer.fit_on_texts(data)
    # Vocabulary size
    vocab_size = len(tokenizer.word_counts) + 1  # Add 1 for the reserved index
    for layer in model.layers:
        if isinstance(layer, Embedding):
            # Retrieve the configuration of the Embedding layer
            embedding_config = layer.get_config()
            # Access the first argument (input_dim) from the configuration
            input_dim = embedding_config['input_dim']
            if input_dim != vocab_size:
                raise ContractException("The first argument provided to the Embedding layer"
                                        " should be the vocabulary size+1.")
            
#excel sheet row 92 
@new_contract
def check_lstm_input_shape(model):
    
    for layer in model.layers:
        if isinstance(layer, LSTM):
            input_shape = layer.input_shape
            if len(input_shape) != 3:
                raise ContractException("Invalid input shape for LSTM layer. "
                                        "Expected input shape (batch_size, timesteps, input_dim), ")
            
#excel sheet row 99
@new_contract
def check_rnn_input_shape(model):
    for layer in model.layers:
        if isinstance(layer, keras.layers.recurrent.RNN):
            # Access the input shape of the RNN layer
            input_shape = layer.input_shape
            if len(input_shape) == 2:
                # Check if num_features are specified
                num_timesteps, num_features = input_shape
                if  num_features is None:
                    raise ContractException("The num_features should be specified in the input shape.")
            elif len(input_shape) == 3:
                # Check if num_features are specified
                num_samples,num_timesteps, num_features = input_shape
                if  num_features is None:
                    raise ContractException("The num_features should be specified in the input shape.")
            else:
                raise ContractException("Invalid input shape for RNN layer. "
                                        "Expected input shape (num_timesteps, num_features) or" 
                                        "(num_samples, num_timesteps, num_features), ")
            
#excel sheet row 103
@new_contract
def check_model_input_shape(model):
    if isinstance(model, Sequential):
        layer=model.layers[0]
        # Check if the input_shape attribute exists and is not None
        if not hasattr(layer, 'input_shape') or layer.input_shape is None:
            raise ContractException("The input shape of the first layer should be explicitly specified.")

#excel sheet row 104
@new_contract
def check_add_method(model):
    """checks whether the add method is present and callable,
       which would indicate that layers have been added to the model using the add method.
    """
    if hasattr(model, 'add') and callable(getattr(model, 'add', None)):
        if not isinstance(model, Sequential):
            raise ContractException("In order to add layers to a model, using add method the model should "
                                     "be a Sequential model. In case of loading a saved model, create a "
                                     "new Sequential model then add loaded model, then you can use add method"
                                      " to add layers") 
        
#excel sheet row 105
@new_contract
def check_load_weights_method(model):
    """
    Check if the model variable has the load_weights method call like model.load_weights().
    Then check if the variable is a Keras Model before trying to load weights.

    Parameters:
    - model: The loaded model object.
    """
    if hasattr(model, 'load_weights') or callable(getattr(model, 'load_weights', None)):
        if not isinstance(model, Model) or len(model.layers) == 0 :
            raise ContractException("In order to load weights to a model,"
                                    "the model should be a Keras Model and have layers added."
                                    "Try to create or load the model before loading weights.")

#excel sheet row 109
@new_contract
def is_custom_loss(model):
    # Get the compile arguments of the model
    compile_args = model.compile_args

    # Check if 'loss' key is present and corresponds to a callable object
    if 'loss' in compile_args and callable(compile_args['loss']):
        return True
    else:
        raise ContractException("If the loss function is a custom loss function, "
                                "then pass it as a function object, not a string.")

#excel sheet row 113: 
@new_contract
def check_confusion_matrix_input(y_test, y_pred):
    """
    Check if the input to confusion_matrix is in the correct format.

    checks if the inputs have more than one dimension, which is a common 
    characteristic of one-hot encoded arrays.
    """
    if y_test.ndim != 1 or y_pred.ndim != 1:
        raise ContractException("Input must be an array of int, not one hot encodings")
    #check ifinput is an integer array
    if not np.issubdtype(y_test.dtype, np.integer) and np.issubdtype(y_pred.dtype, np.integer):
        raise ContractException("Input must be an array of int .")
    
#excel sheet row 114:
@new_contract
def check_evaluate_assignment(result):
    """
    Check if the user has assigned a pair of variables to the result of 'model.evaluate'.
    """
    if isinstance(result, tuple) and len(result) > 1:
        raise ContractException("Assigning a pair of variables to the result of 'model.evaluate' is not allowed. "
                                "Use a single variable to capture the result.")

#excel sheet row 121
@new_contract
def check_input_shape(model, X):
    """
    Check if the input data shape matches the expected input shape of the model.

    Parameters:
        - model: The Keras model.
        - X: The input data.

    Raises:
        - ContractException: If the input data shape does not match the expected input shape.
    """
    expected_input_shape = model.input_shape[1:]  # Exclude batch size
    actual_input_shape = X.shape[1:]

    if expected_input_shape != actual_input_shape:
        raise ContractException(f"Input data shape {actual_input_shape} does not match "
                                f"the expected input shape {expected_input_shape} of the model.")

#excel sheet row 126
@new_contract
def check_normalized_input(data):
    """
    Check if the input data is properly normalized.

    """
    # Check if the data is a numpy array
    if not isinstance(data, np.ndarray):
        raise ContractException("Input data must be a NumPy array.")
    
    # Check if the data is in the range [0, 1]
    if not (np.min(data) >= 0 and np.max(data) <= 1):
        raise ContractException("Input data must be normalized between 0 and 1.")
    
#excel sheet row 143
@new_contract
def check_build_fn_parameter(classifier: KerasClassifier):
    """
    Check the build_fn parameter of a KerasClassifier.
    """
    build_fn = classifier.build_fn

    if not callable(build_fn):
        raise ContractException("build_fn must be a callable (function or class instance).")

    if inspect.isclass(build_fn):
        # If build_fn is a class, check if it has a __call__ method
        if not hasattr(build_fn, '__call__'):
            raise ContractException("If build_fn is a class, it must have a __call__ method.")

        # Check if the instance of the class returns a compiled Keras model
        instance = build_fn()
        if not isinstance(instance, Model):
            raise ContractException("The __call__ method of the build_fn class must return a compiled Keras model.")
    elif not isinstance(build_fn(), Model):
        raise ContractException("The build_fn function must return a compiled Keras model.")

#excel sheet row 145
@new_contract
def check_classifier_is_trained(classifier: KerasClassifier):
    """
    Check if the 'fit' method has been called on the KerasClassifier object.

    This contract checks if the model_ attribute exists in the classifier object, and if it's None, 
    then it raises a ContractException. This is a simple way to ensure that the fit method has been
    called on the KerasClassifier object
    """
    if not hasattr(classifier, 'model_') or classifier.model_ is None:
        raise ContractException("The 'fit' method must be called on the classifier before making predictions.")
       
    
#excel sheet row 240 
@new_contract
def check_distinct_layer_names(model1, model2):
    """
    Check if the two Keras models have distinct layer names.
    """
    layer_names_1 = set(layer.name for layer in model1.layers)
    layer_names_2 = set(layer.name for layer in model2.layers)

    common_layer_names = layer_names_1.intersection(layer_names_2)

    if common_layer_names:
        raise ContractException(f"Common layer names found between the two models: {common_layer_names}. "
                                "Ensure that each model has distinct layer names before concatenating them.")

#excel sheet row 142
@contract
def check_same_tokenizer(tokenizer_train: Tokenizer, tokenizer_test: Tokenizer):
    """
    Check if the same tokenizer has been used for the training and test datasets.
    """
    if tokenizer_train.word_index != tokenizer_test.word_index:
        raise ContractException("Different tokenizers used for training and test datasets.")

#excel sheet row 6
@new_contract
def check_gradient(grad):
    """
    check if grad(Gradient) is none, if its none will raise contract exception
    Parameters:
        grad (Tensor): The loss tensor.
    """
    if grad is None:
        raise ContractException("To properly compute the gradient of the loss with respect to the input,"
        " you need to use model.input")

#excel sheet row 9
@new_contract
def check_add_method(model):
    """
    Check if the user is trying to add layers to the model using the add() method.

    Parameters:
        model (Model): The Keras model.
        
    Raises:
        ContractException: If the user is trying to add layers to the model using the add() method.
    """
    if not isinstance(model, Sequential):
        if hasattr(model, 'add') and callable(getattr(model, 'add')):
            raise ContractException("Cannot add layers directly to the model that is not sequential using the add() method. "
                                "Please use the Functional API to create a new model with the desired architecture.")

#excel sheet row 23
@new_contract
def check_shape_mismatch(model, input_data):
    """
    Check for shape mismatch between expected input shapes of all layers in the model
    and the actual input data shape passed to each layer during training or evaluation.

    Parameters:
        model: The Keras model for which input shapes are checked.
        input_data: The input data passed to the model.

    Raises:
        ContractException: If there is a shape mismatch between expected input shape and actual input data shape
                           for any layer in the model.
    """
    for layer in model.layers:
        expected_shape = layer.input_shape
        if expected_shape is not None:
            actual_shape = input_data.shape

            if expected_shape[1:] != actual_shape[1:]:
                raise ContractException("There is mismatch between the expected input shape of layer and actual data input shape"
                                        "If you are trying to reuse the layer from another model, you should define it as a separate layer object")

#excel sheet row 25
@new_contract
def check_load_weights_assignment(model):
    """
    Check if the variable being assigned to load_weights is a Keras model.

    Parameters:
        model: The variable being assigned to load_weights.

    Raises:
        ContractException: If the variable is a Keras model, suggesting to use load_model instead of load_weights.
    """
    if isinstance(model, Model):
        raise ContractException("The variable being assigned to load_weights is a Keras model. "
                                "Please use load_model instead of load_weights to load a complete model.")

#excel sheet row 85
@new_contract
def check_lstm_complexity(model, input_shape, num_time_frames):
    """
    Check if the model contains complex LSTM layers combined with large input data shape and number of time frames,
    and suggest replacing them with CuDNNLSTM layers.
    input_shape[0] represents the number of samples in the input data, and input_shape[1] represents the number of features per sample.

    Parameters:
        model (Sequential): The Keras model.
        input_shape (tuple): The shape of the input data.
        num_time_frames (int): The number of time frames in the input data.

    Raises:
        ContractException: If the model contains complex LSTM layers combined with large input data shape
                           and number of time frames.
    """
    max_lstm_units = 50  # Define the threshold for a high number of LSTM units
    max_input_shape = 10000  # Define the threshold for a large input data shape
    max_time_frames = 10  # Define the threshold for a large number of time frames

    num_lstm_layers = 0

    for layer in model.layers:
        if isinstance(layer, LSTM):
            num_lstm_layers += 1
            if (layer.units > max_lstm_units) or (input_shape[0] * input_shape[1] > max_input_shape) or \
               (num_time_frames > max_time_frames):
                raise ContractException("Consider replacing complex LSTM layers with CuDNNLSTM layers "
                                        "for improved training time.")

#excel sheet row 86
@new_contract
def check_input_shape(model: Sequential, X: Sequence):
    """
    Check if the input shape of the model matches the shape of the input data.

    Parameters:
        model (Sequential): The Keras model.
        X (Sequence): The input data.

    Raises:
        ContractException: If the input shape of the model does not match the shape of the input data.
    """
    if isinstance(model, Sequential):
        input_shape = model.input_shape
        if input_shape[1:] != X.shape[1:]:
            raise ContractException("Input shape of the model ({}) does not match the shape of the input data ({})".format(input_shape[1:], X.shape[1:]))

    
#excel sheet row 87
@new_contract
def get_keras_layer_weights(layer):
    """
    Check if the object has get_weights attribute then check if it is a Keras layer.

    Parameters:
        layer: keras.layers.Layer

    Returns:
        tuple: A tuple containing the layer's weights (numpy arrays) and biases, 
               or None if the layer has no weights.

    Raises:
        ValueError: If the input is not a Keras layer.
    """
    if hasattr(layer, 'get_weights') and callable(layer.get_weights):
        if not isinstance(layer, keras.layers.Layer):
            ContractException("Input must be a Keras layer to apply get_weights() attribute")
    

#excel sheet row 90
@new_contract
def check_flatten_layer_usage(layer):
    """
    Check whether the Flatten layer is instantiated and called correctly.

    Parameters:
        layer: The Flatten layer instance.

    Raises:
        ContractException: If the Flatten layer is not instantiated and called correctly.
    """
    if isinstance(layer, Flatten):
        if not callable(layer):
            raise ContractException("The Flatten layer must be instantiated then called as a function with the tensor as an argument.")   
        

#excel sheet row 96
@new_contract
def check_numpy_operations_para(para1, para2):
    """
    Check if numpy operations are being used on parameters that are not numpy arrays.

    Parameters:
        para1, para2 : numpy operator parameters

    Raises:
        ContractException: If numpy operations are used on non-numpy arrays.
    """
    # Check if y_true and y_pred are numpy arrays
    if not isinstance(para1, np.ndarray) or not isinstance(para2, np.ndarray):
        raise ContractException("Parameters must be numpy arrays. Numpy operations can only be applied to numpy arrays")

#excel sheet row 98
@new_contract
def check_concatenation_parameters(parameters):
    # Check if all parameters have the same data type
    data_types = [K.dtype(p) for p in parameters]
    if len(set(data_types)) != 1:
        raise ContractException("Parameters have different data types.")
    
    # Check if shapes are compatible
    shapes = [K.int_shape(p) for p in parameters]
    axis = -1  # Default axis for concatenation
    for shape in shapes:
        if shape[axis] is None:
            raise ContractException("One or more parameters have undefined shape along concatenation axis.")

    if any(shape[axis] != shapes[0][axis] for shape in shapes):
        raise ContractException("Shapes along concatenation axis do not match.")

#excel sheet row 107
@new_contract
def check_custom_loss(model):
    # Get the compile arguments of the model
    compile_args = model.compile_args

    # Check if 'loss' key is present and corresponds to a callable object
    if 'loss' in compile_args and callable(compile_args['loss']) is None:
        raise ContractException("Your custom loss function is not properly defined, "
                                "It is returning None")

#excel sheet row 111
@new_contract
def matching_input_order(model, data):
    """
    Contract to check if the input shapes have matching ordering.

    Parameters:
        model: keras model containing the layers.
        data: Input data.

    Raises:
        ContractException: If the ordering of input shapes does not match.
    """
    for layer in model.layers:
        if layer.input_shape[-1] != data.shape[-1]:
            raise ContractException("Mismatched input ordering: "
                                "Layer input shape {layer.name} and data input shape  "
                                "have different channel ordering.")

#excel sheet row 112
@new_contract
def validate_cross_val_score():
    """
    We use kwargs.get('fit_params') to get the value of the 'fit_params' argument from the function call.
    """
    if 'fit_params' in inspect.signature(cross_val_score).parameters:
        if not isinstance(kwargs.get('fit_params'), dict):
            raise ContractException("fit_params argument is present and not of dictionary type."
                                    "where the keys are parameter names and the values are lists of "
                                    "parameters to pass to the fit method ")


#excel sheet row 122
@new_contract
def numpy_array(X):
    """
    Contract to check if the input data passed to the fit method is a NumPy array.

    Parameters:
        X: Input data passed to the fit method.

    Raises:
        ContractException: If the input data is not a NumPy array.
    """
    if not isinstance(X, ndarray):
        raise ContractException("Input data must be a NumPy array.")


#excel sheet row 128
@new_contract
def validate_history(history):
    """
    Contract to validate the training history recorded by Keras.

    Parameters:
        history: History object returned by model.fit().

    Raises:
        ContractException: If the training history is empty.
    """
    if not history.history:
        raise ContractException("Training history is empty. Check if the object has been passed"
        " as a callback to the model.fit() method.")

# excel sheet row 131
@new_contract
def accessible_custom_objects(model_file, custom_objects):
    """
    Contract to check if custom objects are accessible by the model.

    Parameters:
        model_file (str): File path to the saved model.
        custom_objects (dict): Dictionary containing custom objects.

    Raises:
        ContractException: If the custom objects are not accessible by the model.
    """
    with CustomObjectScope(custom_objects):
        try:
            model = load_model(model_file)
            # If the model is loaded successfully, custom objects are accessible
        except Exception as e:
            raise ContractException(f"Custom objects are not accessible: {e}")