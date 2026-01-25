from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.annotations import override, OldAPIStack
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.models.torch.misc import SlimFC

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# ========================================================================
# ### NEW PYTORCH MODEL (FOR DICT OBSERVATIONS) ###
# ========================================================================
class CustomConv3DModel(TorchModelV2, nn.Module):
    """
    Multi-modal network for Dict observation space:
    - CNN branch for 3D "dose" input.
    - MLP branch for 1D "beams" input.
    - Concatenates features for policy and value heads.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        

        # --- We must handle a Dict obs_space ---
        
        # 1. Define the CNN branch for the "dose" input
        
        # --- START FIX ---
        # The 'obs_space' object IS the Dict. Access its keys directly.
        dose_obs_shape = obs_space["dose"].shape
        # --- END FIX ---
            
        # dose_obs_shape[0] is the channel count (2)
        
        self.cnn_branch = nn.Sequential(
            nn.Conv3d(dose_obs_shape[0], 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened size from the CNN
        with torch.no_grad():
            sample_input = torch.zeros((1,) + dose_obs_shape)
            cnn_out_size = self.cnn_branch(sample_input).shape[1]

        # Add a fully connected layer just for the CNN features
        self.cnn_fc = SlimFC(cnn_out_size, 256, activation_fn="relu")
        
        # 2. Define the MLP branch for the "beams" input
        
        # --- START FIX ---
        # Access this key directly as well
        beams_obs_size = obs_space["beams"].shape[0]
        # --- END FIX ---
        
        self.beams_mlp = SlimFC(beams_obs_size, 128, activation_fn="relu")

        # 3. Define the combined "heads"
        
        # Calculate the total size of concatenated features
        combined_features_size = 256 + 128

        # Policy head (for action logits)
        self.policy_head = SlimFC(combined_features_size, num_outputs)

        # Value head (for value function)
        self._value_head = SlimFC(combined_features_size, 1)
        
        # To store features for value function
        self._features = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # --- Process the Dict observation ---
        
        # Extract the two inputs from the observation dictionary
        dose_input = input_dict["obs"]["dose"]
        beams_input = input_dict["obs"]["beams"]

        # Process dose through CNN
        cnn_out = self.cnn_branch(dose_input)
        cnn_features = self.cnn_fc(cnn_out)
        
        # Process beams through MLP
        beams_features = self.beams_mlp(beams_input)

        # Concatenate the features from both branches
        self._features = torch.cat([cnn_features, beams_features], dim=1)

        # Compute action logits
        logits = self.policy_head(self._features)
        
        return logits, state

    @override(ModelV2)
    def value_function(self):
        # --- Use the combined features ---
        assert self._features is not None, "must call forward() first"
        return self._value_head(self._features).squeeze(1)


# ========================================================================
# ### OLD TENSORFLOW MODEL (UNMODIFIED) ###
# ========================================================================
class ExpanseLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpanseLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.expand_dims(x, axis=-1)

@OldAPIStack 
class CustomConv3DModelTF(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomConv3DModelTF, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # ---
        # NOTE: This model will NOT work with the new Dict observation space.
        # It would need to be rewritten like the PyTorch model above.
        # ---
        if not isinstance(obs_space, gym.spaces.Box):
             raise NotImplementedError(
                 f"This TF model only supports Box obs space, "
                 f"but got {obs_space}. Update this model to handle Dicts."
            )

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
            kernel_initializer=normc_initializer(1.0)
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