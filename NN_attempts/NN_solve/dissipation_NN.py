import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List, Tuple


class Conv2D(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: str

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding='SAME', key=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_channels, in_channels, kernel_size[0], kernel_size[1])) * 0.02
        self.bias = jax.random.normal(bkey, (out_channels,)) * 0.02

    def __call__(self, x):
        return jax.lax.conv_general_dilated(
            x, 
            self.weight, 
            window_strides=self.stride,
            padding=self.padding
        ) + self.bias.reshape(1, -1, 1, 1)


class LSTMCell(eqx.Module):
    hidden_size: int
    input_size: int
    weight_ih: jnp.ndarray
    weight_hh: jnp.ndarray
    bias_ih: jnp.ndarray
    bias_hh: jnp.ndarray

    def __init__(self, input_size, hidden_size, key=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        wkey1, wkey2, bkey1, bkey2 = jax.random.split(key, 4)
        
        self.weight_ih = jax.random.normal(wkey1, (4 * hidden_size, input_size)) * 0.02
        self.weight_hh = jax.random.normal(wkey2, (4 * hidden_size, hidden_size)) * 0.02
        self.bias_ih = jax.random.normal(bkey1, (4 * hidden_size,)) * 0.02
        self.bias_hh = jax.random.normal(bkey2, (4 * hidden_size,)) * 0.02

    def __call__(self, h_c, x):
        h, c = h_c
        gates = (jnp.matmul(x, self.weight_ih.T) + self.bias_ih +
                jnp.matmul(h, self.weight_hh.T) + self.bias_hh)
        
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        
        c = f * c + i * g
        h = o * jnp.tanh(c)
        
        return (h, c)


class LSTM(eqx.Module):
    cells: List[LSTMCell]
    
    def __init__(self, input_size, hidden_size, num_layers=1, key=None):
        keys = jax.random.split(key, num_layers)
        self.cells = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, key=keys[i])
            for i in range(num_layers)
        ]
    
    def __call__(self, inputs, initial_state=None):
        batch_size, seq_len, _ = inputs.shape
        
        if initial_state is None:
            initial_state = [(jnp.zeros((batch_size, cell.hidden_size)), 
                             jnp.zeros((batch_size, cell.hidden_size)))
                             for cell in self.cells]
        
        # Process sequence step by step
        def scan_fn(carry, x):
            states = carry
            for i, cell in enumerate(self.cells):
                states[i] = cell(states[i], x)
                x = states[i][0]
            return states, x
        
        # Run over sequence
        states, outputs = jax.lax.scan(
            scan_fn,
            initial_state,
            jnp.transpose(inputs, (1, 0, 2))  # [seq_len, batch, features]
        )
        
        # Return last hidden state and outputs
        return outputs[-1], states


class DissipationNet(eqx.Module):
    conv1: Conv2D
    conv2: Conv2D
    lstm: LSTM
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc_out: eqx.nn.Linear
    speed_scale: eqx.nn.Linear

    def __init__(self, input_channels, seq_length, hidden_dim=64, key=None):
        keys = jax.random.split(key, 6)
        
        self.conv1 = Conv2D(input_channels, 32, kernel_size=(3, 3), padding='SAME', key=keys[0])
        self.conv2 = Conv2D(32, 64, kernel_size=(3, 3), padding='SAME', key=keys[1])
        
        self.lstm = LSTM(input_size=64, hidden_size=hidden_dim, num_layers=2, key=keys[2])
        
        self.fc1 = eqx.nn.Linear(hidden_dim, 32, key=keys[3])
        self.fc2 = eqx.nn.Linear(32, 16, key=keys[4])
        self.fc_out = eqx.nn.Linear(16, 1, key=keys[5])
        
        # Initialize scale with physical prior - no bias for physical scaling
        self.speed_scale = eqx.nn.Linear(1, 1, use_bias=False)
        # Initialize to ~1.0 for proper |U|³ scaling
        self.speed_scale.weight = jnp.array([[1.0]])

    def __call__(self, x, u_mag):
        # x: [batch, channels, height, width, time]
        batch_size, C, H, W, T = x.shape
        
        # Process each timestep through CNN
        cnn_out = []
        for t in range(T):
            xt = x[:, :, :, :, t]  # [batch, C, H, W]
            xt = jax.nn.relu(self.conv1(xt))
            xt = jax.lax.reduce_window(xt, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID')
            
            xt = jax.nn.relu(self.conv2(xt))
            xt = jnp.mean(xt, axis=(2, 3))  # Global average pooling
            cnn_out.append(xt)
        
        # Combine timesteps
        cnn_out = jnp.stack(cnn_out, axis=1)  # [batch, seq, features]
        
        # Process temporal dependencies
        lstm_out, _ = self.lstm(cnn_out)
        
        # Fully connected layers
        x = jax.nn.relu(self.fc1(lstm_out))
        x = jax.nn.relu(self.fc2(x))
        dissipation_factor = self.fc_out(x)
        
        # Physical scaling - ensure dissipation scales with |U|³
        u_mag_cubed = u_mag ** 3
        dissipation = jax.nn.softplus(dissipation_factor) * u_mag_cubed
        
        return dissipation