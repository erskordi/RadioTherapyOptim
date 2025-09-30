from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.annotations import override, OldAPIStack
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.models.torch.misc import SlimFC

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class CustomConv3DModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Example Conv3D layers
        self.conv = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened size
        with torch.no_grad():
            sample_input = torch.zeros((1, 2, 129, 167, 167))
            conv_out_size = self.conv(sample_input).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            SlimFC(conv_out_size, 256, activation_fn="relu"),
            SlimFC(256, num_outputs)
        )

        self._value_branch = SlimFC(256, 1)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]  # Add channel dimension
        conv_out = self.conv(obs)
        features = self.fc[0](conv_out)
        logits = self.fc[1](features)
        self._features = features  # store for value function
        return logits, state

    @override(ModelV2)
    def value_function(self):
        return self._value_branch(self._features).squeeze(1)

class ExpanseLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpanseLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.expand_dims(x, axis=-1)

@OldAPIStack 
class CustomConv3DModelTF(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomConv3DModelTF, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        input_shape = obs_space.shape  # (depth, height, width)

        # Input layer (channels_last format: D, H, W, C)
        inputs = tf.keras.Input(shape=input_shape, name="observations")
        x = ExpanseLayer()(inputs)  # (D, H, W, 1)

        # Conv3D layers
        x = tf.keras.layers.Conv3D(
            filters=8,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
            kernel_initializer="he_normal"
        )(x)

        x = tf.keras.layers.Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
            kernel_initializer="he_normal"
        )(x)

        x = tf.keras.layers.Flatten()(x)

        # Fully connected (hidden layer)
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_initializer=normc_initializer(1.0)  # RLlib prefers normc_initializer for stable RL training
        )(x)

        # Output layer for policy logits
        logits = tf.keras.layers.Dense(
            num_outputs,
            activation=None,
            kernel_initializer=normc_initializer(0.01)
        )(x)

        # Separate value branch (for value function)
        value_out = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=normc_initializer(0.01)
        )(x)

        # Save Keras model
        self.base_model = tf.keras.Model(inputs, [logits, value_out])
        #self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}