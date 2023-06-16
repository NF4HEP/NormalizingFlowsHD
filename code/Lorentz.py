import json
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import LambdaCallback
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from corner import corner
import Utils

class BaseDistribution4Momenta(tfd.Distribution):
    """
    A base distribution for a physical system of n_particles in 4-momenta representation.
    """
    def __init__(self, 
                 n_particles, 
                 masses=None, 
                 means3mom=None, 
                 stdev3mom=None,
                 conserve_transverse_momentum=True,
                 name="BaseDistribution4Momenta", 
                 dtype=tf.float32, 
                 reparameterization_type=tfd.FULLY_REPARAMETERIZED, 
                 validate_args=False, 
                 allow_nan_stats=True, 
                 **kwargs):
        """
        Initialize the distribution.
        
        Parameters
        ----------
        n_particles : int, strictly positive
            Number of particles in the system.
        masses : array-like, optional, default None, shape (n_particles,)
            Masses of the particles. If None, all masses are set to zero.
        means3mom : array-like, optional, default None, shape (3,)
            Means of the 3-momenta of the particles. If None, all means are set to zero.
        stdev3mom : array-like, optional, default None, shape (3,)
            Standard deviations of the 3-momenta of the particles. If None, all standard deviations are set to one.
        conserve_transverse_momentum : bool, optional, default True
            Flag that determines if the base distribution should conserve the transverse momentum.
        name : str, optional, default 'BaseDistribution4Momenta'
            Name of the distribution.
        dtype : tf.dtype, optional, default tf.float32
            Type of the distribution.
        reparameterization_type : tfp.distributions.ReparameterizationType, optional, default tfd.FULLY_REPARAMETERIZED
            Reparameterization type of the distribution.
        validate_args : bool, optional, default False
            Whether to validate input with asserts.
        allow_nan_stats : bool, optional, default True
            Whether to allow NaN statistics.
        **kwargs
            Additional keyword arguments.
            
        Raises
        ------
        ValueError
            If the number of particles is not positive.
        ValueError
            If the number of particles is not an integer.
        ValueError
            If the masses are not positive.
        ValueError
            If the means of the 3-momenta are not finite. 
        ValueError
            If the standard deviations of the 3-momenta are not positive.
        ValueError
            If the standard deviations of the 3-momenta are not finite.

        Returns 
        ------- 
        None    
        """
        super(BaseDistribution4Momenta, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            **kwargs
        )

        self.n_particles = n_particles
        self.transverse_momentum_conservation = conserve_transverse_momentum

        if masses is None:
            masses = tf.zeros((n_particles,),dtype=self.dtype)
        self.masses = tf.cast(masses, dtype=self.dtype)

        if means3mom is None:
            means3mom = tf.zeros((3,), dtype=self.dtype)
        self.means3mom = tf.cast(means3mom, dtype=self.dtype)

        if stdev3mom is None:
            stdev3mom = tf.ones((3,), dtype=self.dtype)
        self.stdev3mom = tf.cast(stdev3mom, dtype=self.dtype)

    def _sample_n(self, n, seed=None):
        """
        Generate n random samples.
        """
        # sample 3-momenta of n_particles particles from normal distribution with mean and stddev given by means3mom and stdev3mom
        samples = tf.random.normal((n, self.n_particles, 3), mean=self.means3mom, stddev=self.stdev3mom, dtype=self.dtype) # (n, n_particles, 3)
        # If transverse momentum conservation is required, fix the last particle's momenta to ensure that the total transverse momentum is zero
        # (px + px_last = 0 and py + py_last = 0)
        if self.transverse_momentum_conservation:
            # compute sum of px and py for each event to use for last particle (px + px_last = 0 and py + py_last = 0)
            sum_px = tf.reduce_sum(samples[..., 0], axis=-1, keepdims=True) # (n, 1)
            sum_py = tf.reduce_sum(samples[..., 1], axis=-1, keepdims=True) # (n, 1)
            # compute px and py for last particle (px_last = -sum_px and py_last = -sum_py)
            samples_fixed_px_py = samples[:, :-1] # (n, n_particles-1, 3)
            # create new tensor with updated last px and py (px_last = -sum_px and py_last = -sum_py) and pz
            last_px_py = -tf.stack([sum_px, sum_py], axis=-1)[:, 0] + samples[:, -1, :2] # (n, 2)
            # extract pz of last particle from samples drawn from normal distribution with mean and stddev given by means3mom and stdev3mom
            last_pz = samples[:, -1, 2]
            # reshape last_px_py and last_pz to match dimensions
            last_px_py = tf.reshape(last_px_py, [n, -1])
            last_pz = tf.reshape(last_pz, [n, -1])
            # create new tensor with updated last px, py and pzs
            last_particle = tf.concat([last_px_py, last_pz], axis=-1)
            samples = tf.concat([samples_fixed_px_py, last_particle[:, tf.newaxis, :]], axis=1)
        # compute energy from momenta and mass
        energy = tf.sqrt(tf.reduce_sum(samples ** 2, axis=-1) + tf.reshape(self.masses, (1, self.n_particles)) ** 2)
        # stack energy and momenta to form 4-momenta
        samples = tf.concat([energy[..., tf.newaxis], samples], axis=-1)
        # reshape the samples to the desired shape
        samples = tf.reshape(samples, [n, 4*self.n_particles])
        return samples

    def _log_prob(self, x):
        """
        Compute log-probability of x under the distribution.
        """
        x = tf.reshape(x, [-1, self.n_particles, 4])
        px = x[..., 1] # shape (n_samples, n_particles)
        py = x[..., 2] # shape (n_samples, n_particles)
        pz = x[..., 3] # shape (n_samples, n_particles)

        if self.transverse_momentum_conservation:
            # only consider px and py for particles other than the last
            px = px[..., :-1] # shape (n_samples, n_particles-1)
            py = py[..., :-1] # shape (n_samples, n_particles-1)

        # normal distribution for momenta
        log_prob_px = tf.reduce_sum(tfd.Normal(self.means3mom[..., 0], self.stdev3mom[..., 0]).log_prob(px), axis=-1)
        log_prob_py = tf.reduce_sum(tfd.Normal(self.means3mom[..., 1], self.stdev3mom[..., 1]).log_prob(py), axis=-1)
        log_prob_pz = tf.reduce_sum(tfd.Normal(self.means3mom[..., 2], self.stdev3mom[..., 2]).log_prob(pz), axis=-1)

        log_prob_sum = log_prob_px + log_prob_py + log_prob_pz

        return log_prob_sum

    def _batch_shape_tensor(self):
        """
        Get the batch shape as a tensor.
        """
        return tf.constant([], dtype=tf.int32)

    def _batch_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return tf.constant([4*self.n_particles], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([4*self.n_particles])


class LorentzTransformNN(tfk.Model):
    """
    A simple neural network that takes a four-momentum as input and outputs the parameters
    of a Lorentz boost that is applied to the four-momentum.
    """
    def __init__(self, 
                 hidden_units, 
                 activation='relu', 
                 name='LorentzTransformNN', 
                 dtype='float32', 
                 **kwargs):
        """
        Initialize the network.
        
        parameters:
        :param hidden_units: The number of units in the hidden layers.
        :param activation: The activation function to use in the hidden layers.
        :param name: The name of the network.
        :param dtype: The data type of the network.
        :param kwargs: Additional arguments for the base class.
        
        return: None
        """
        super(LorentzTransformNN, self).__init__(name=name, **kwargs)
        self.dense_layers = [tfk.layers.Dense(units, activation=activation, kernel_initializer='glorot_uniform', bias_initializer='zeros', dtype=dtype) for units in hidden_units] # Hidden layers
        self.dense_beta = tfk.layers.Dense(1, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros', dtype=dtype) # (boost parameters β_i), ranges from -1 to 1
        self.dense_angles = tfk.layers.Dense(5, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros', dtype=dtype) # (rotation angles theta_i), ranges from 0 to 1 (0 to 2π)

    def call(self, inputs):
        """
        Apply the network to the inputs.
        :param inputs: The inputs. shape: (n_events, n_particles, 4)
        :return: The outputs.
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        beta = self.dense_beta(x)/2+0.5  # scale the output of tanh to [0, 1]
        #betas = 2*self.dense_betas(x)-1  # scale the output of sigmoid to [-1, 1]
        angles = 2 * np.pi * self.dense_angles(x)  # scale the output of sigmoid to [0, 2π]
        parameters = tf.concat([beta, angles], axis=-1)
        return parameters
    
    
class MomentumCorrectionBijector(tfb.Bijector):
    """ 
    A bijector that corrects the momentum of the first particle in an event to conserve momentum.
    """
    def __init__(self, 
                 validate_args=False, 
                 name='MomentumCorrectionBijector'):
        super(MomentumCorrectionBijector, self).__init__(validate_args=validate_args, forward_min_event_ndims=1, name=name)

    def _forward(self, x):
        # Assume the input is an array of shape (n_events, 4*n_particles) where the first 4 values are the
        #  4-momenta of the first particle, the next 4 values are the 4-momenta of the second particle, etc.
        # The 4-momenta are ordered as (E, px, py, pz).
        # The output is an array of the same shape, but with the momentum of the first particle corrected
        #  to conserve momentum.
        
        # Reshape the input to (n_events, n_particles, 4)
        n_particles = x.shape[1] // 4
        x_reshaped = tf.reshape(x, [-1, n_particles, 4])
        
        # Compute the total px and py for particles 2 to N
        total_px = tf.reduce_sum(x_reshaped[:, 1:, 1], axis=1, keepdims=True)
        total_py = tf.reduce_sum(x_reshaped[:, 1:, 2], axis=1, keepdims=True)
        
        # Subtract these from the first particle to conserve momentum
        corrected_px = -total_px
        corrected_py = -total_py
        
        # Compute the energy using the on-shell condition (E^2 = m^2 + p^2)
        original_mass_squared = x_reshaped[:, 0, :1] ** 2 - tf.reduce_sum(x_reshaped[:, 0, 1:] ** 2, axis=-1, keepdims=True)
        corrected_energy = tf.sqrt(original_mass_squared + corrected_px ** 2 + corrected_py ** 2 + x_reshaped[:, 0, 3:4] ** 2)
        
        # Create a new array with the corrected particle and the transformed particles
        corrected_particle = tf.concat([corrected_energy, corrected_px, corrected_py, x_reshaped[:, :1, 3]], axis=1)

        # Concatenate the corrected particle with the transformed particles
        y = tf.concat([corrected_particle[:, None, :], x_reshaped[:, 1:, :]], axis=1)
        
        # Reshape the output to (n_events, 4*n_particles)
        y = tf.reshape(y, [-1, 4*n_particles])

        return y

    def _inverse(self, y):
        # Assume the input is an array of shape (n_events, 4*n_particles) where the first 4 values are the
        #  4-momenta of the first particle, the next 4 values are the 4-momenta of the second particle, etc.
        # The 4-momenta are ordered as (E, px, py, pz).
        # The output is an array of the same shape, but with the momentum of the first particle corrected
        #  back to its original value
        
        # Reshape the input to (n_events, n_particles, 4)
        n_particles = y.shape[1] // 4
        y_reshaped = tf.reshape(y, [-1, n_particles, 4])
        
        # Compute the total px and py for particles 2 to N
        total_px = tf.reduce_sum(y_reshaped[:, 1:, 1], axis=1, keepdims=True)
        total_py = tf.reduce_sum(y_reshaped[:, 1:, 2], axis=1, keepdims=True)
        
        # Add these to the first particle to get back the original momentum
        original_px = total_px
        original_py = total_py
        
        # Compute the energy using the on-shell condition (E^2 = m^2 + p^2)
        original_mass_squared = y_reshaped[:, 0, :1] ** 2 - tf.reduce_sum(y_reshaped[:, 0, 1:] ** 2, axis=-1, keepdims=True)
        original_energy = tf.sqrt(original_mass_squared + original_px ** 2 + original_py ** 2 + y_reshaped[:, 0, 3:4] ** 2)
        
        # Create a new array with the original particle and the transformed particles
        original_particle = tf.concat([original_energy, original_px, original_py, y_reshaped[:, :1, 3]], axis=1)

        # Concatenate the original particle with the transformed particles
        x = tf.concat([original_particle[:, None, :], y_reshaped[:, 1:, :]], axis=1)
        
        # Reshape the output to (n_events, 4*n_particles)
        x = tf.reshape(x, [-1, 4*n_particles])

        return x
    
    def _forward_log_det_jacobian(self, x):
        """
        The Jacobian determinant of a momentum correction is 1
        """
        output = tf.constant(0., x.dtype)
        return output
    
    def _inverse_log_det_jacobian(self, y):
        """
        The Jacobian determinant of a momentum correction is 1
        """
        output = tf.constant(0., y.dtype)
        return output


class GeneralLorentzTransformBijector(tfb.Bijector):
    """
    A bijector that performs a general Lorentz transformation on a 4-momentum vector.
    """
    def __init__(self, 
                 #n_particles, 
                 lorentz_transform_NN,
                 validate_args=False, 
                 name='GeneralLorentzTransformBijector'):
        """
        A bijector that performs a general Lorentz transformation on a 4-momentum vector.
        :param n_particles: The number of particles in the event.
        :param lorentz_transform_NN: A function that takes a 4-momentum vector and returns the parameters of the Lorentz transformation.
        :param validate_args: Python `bool` indicating whether arguments should be checked for correctness.
        :param name: Python `str`, name given to ops managed by this object.
        """
        super(GeneralLorentzTransformBijector, self).__init__(
            is_constant_jacobian=True,
            validate_args=validate_args, 
            forward_min_event_ndims=1, 
            name=name)
        #self.n_particles = n_particles
        self.lorentz_transform_NN = lorentz_transform_NN
        x = tf.keras.Input((4,))
        parameters = self.lorentz_transform_NN(x)
        self.nn = Model(x, parameters)
        
    def _forward(self, x):
        """
        Perform a general Lorentz transformation on a 4-momentum vector.
        :param x: The 4-momentum vector.
        :return: The transformed 4-momentum vector.
        """
        x_a = x[:, :4]  # shape: (n_samples, 4)
        x_b = x[:, 4:]  # shape: (n_samples, 4*(n_particles-1))
        # Use the LorentzTransformNN to choose the Lorentz transformation
        parameters = self.nn(x_a)  # shape: (n_samples, 6)
        beta = parameters[..., 0]  # Velocity for boost in the x direction (ranges from -1 to 1). shape (n_samples,)
        theta = parameters[..., 1]  # Polar angle for boost in the x direction (ranges from 0 to pi). shape (n_samples,)
        phi = parameters[..., 2]  # Azimuthal angle for boost in the x direction (ranges from 0 to 2*pi). shape (n_samples,)
        theta_1 = parameters[..., 3]  # Euler angle 1: rotation around z-axis. shape (n_samples,)
        theta_2 = parameters[..., 4]  # Euler angle 2: rotation around y-axis. shape (n_samples,)
        theta_3 = parameters[..., 5]  # Euler angle 3: rotation around new z-axis. shape (n_samples,)
        # Apply the chosen Lorentz transformation to all particles
        y_a = x_a
        y_b = lorentz_transform(x_b, beta, theta, phi, theta_1, theta_2, theta_3, dtype=x.dtype)
        y = tf.concat([y_a, y_b], axis=-1)
        return tf.reshape(y, tf.shape(x))
    
    def _inverse(self, y):
        """
        Perform the inverse of a general Lorentz transformation on a 4-momentum vector.
        
        Parameters:
        :param y: The 4-momentum vector.
        
        Return:
        :return: The transformed 4-momentum vector.
        """
        y_a = y[:, :4]  # shape: (n_samples, 4)
        y_b = y[:, 4:]  # shape: (n_samples, 4*(n_particles-1))
        # Use the LorentzTransformNN to choose the Lorentz transformation
        parameters = self.nn(y_a)  # shape: (n_samples, 6)
        beta = parameters[..., 0]  # Velocity for boost in the x direction (ranges from -1 to 1). shape (n_samples,)
        theta = parameters[..., 1]  # Polar angle for boost in the x direction (ranges from 0 to pi). shape (n_samples,)
        phi = parameters[..., 2]  # Azimuthal angle for boost in the x direction (ranges from 0 to 2*pi). shape (n_samples,)
        theta_1 = parameters[..., 3]  # Euler angle 1: rotation around z-axis. shape (n_samples,)
        theta_2 = parameters[..., 4]  # Euler angle 2: rotation around y-axis. shape (n_samples,)
        theta_3 = parameters[..., 5]  # Euler angle 3: rotation around new z-axis. shape (n_samples,)
        # Apply the chosen Lorentz transformation to all particles
        x_a = y_a
        x_b = inverse_lorentz_transform(y_b, beta, theta, phi, theta_1, theta_2, theta_3, dtype=y.dtype)
        x = tf.concat([x_a, x_b], axis=-1)
        return tf.reshape(x, tf.shape(y))
    
    def _forward_log_det_jacobian(self, x):
        """
        The Jacobian determinant of a Lorentz transformation is 1
        """
        output = tf.constant(0., x.dtype)
        return output
    
    def _inverse_log_det_jacobian(self, y):
        """
        The Jacobian determinant of a Lorentz transformation is 1
        """
        output = tf.constant(0., y.dtype)
        return output

class GeneralLorentzNormalizingFlow(tfb.Chain):
    """
    A normalizing flow that applies a general Lorentz transformation to each particle in an event.
    """
    def __init__(self, 
                 n_particles, 
                 n_bijectors, 
                 hidden_units, 
                 activation='relu',
                 conserve_transverse_momentum=True,
                 name='ParticleBoostedNormalizingFlow', 
                 dtype="float32",
                 permute_only=False, 
                 **kwargs):
        """
        A normalizing flow that applies a general Lorentz transformation to each particle in an event.
        :param n_particles: The number of particles in the event.
        :param n_bijectors: The number of bijectors in the flow.
        :param hidden_units: The number of hidden units in the Lorentz transformation selector.
        :param conserve_transverse_momentum: Python `bool` indicating whether to conserve transverse momentum in the Lorentz transformation selector.
        :param activation: The activation function to use in the Lorentz transformation selector.
        :param name: Python `str`, name given to ops managed by this object.
        :param permute_only: Python `bool` indicating whether to only permute the particles. Only useful for debugging paricles permuted by the flow.
        :param kwargs: Additional keyword arguments passed to the `tfk.Model` superclass.
        """
        # If n_bijectors < n_particles, set n_bijectors to n_particles and print a warning
        if n_bijectors < n_particles:
            print("Warning: Number of bijectors was less than the number of particles. Performances may be poor.")
            #print("Warning: Number of bijectors was less than the number of particles. Number of bijectors has been set to the number of particles.")
            #n_bijectors = n_particles
        
        self.n_particles = n_particles # The number of particles in the event
        self.n_bijectors = n_bijectors # The number of bijectors in the flow
        self.hidden_units = hidden_units  # The number of hidden units in the Lorentz transformation selector
        self.conserve_transverse_momentum = conserve_transverse_momentum # Python `bool` indicating whether to conserve transverse momentum in the Lorentz transformation selector
        self.activation = activation # The activation function to use in the Lorentz transformation selector
        self.flow_dtype = dtype # The dtype of the flow
        bijectors = [] # The list of bijectors that make up the flow
        lorentz_transform_NNs = [] # The list of Lorentz transformation selectors

        # Base permutation order
        base_perm = np.roll(np.arange(4*n_particles), -4)
        
        for i in range(self.n_bijectors):
            # Add a Permute bijector to the list of bijectors
            permutation = tfb.Permute(permutation=base_perm)

            # If permute_only is True, only add the Permute bijector to the list of bijectors
            # Otherwise, add a LorentzTransformNN and a GeneralLorentzTransformBijector to the list of bijectors
            if permute_only:
                # Add a Permute bijector to the list of bijectors
                bijector = tfb.Identity()
                bijectors.extend([bijector, permutation])
            else:
                # Initialize a LorentzTransformNN to choose the Lorentz transformation
                lorentz_transform_NN = LorentzTransformNN(hidden_units=self.hidden_units, activation=self.activation, name='LorentzTransformNN_'+str(i), dtype=self.flow_dtype)
                lorentz_transform_NNs.append(lorentz_transform_NN)
                # Add a GeneralLorentzTransformBijector to the list of bijectors
                bijector = GeneralLorentzTransformBijector(lorentz_transform_NN=lorentz_transform_NN, name='GeneralLorentzTransformBijector_'+str(i))
                if self.conserve_transverse_momentum:
                    # If transverse momentum is conserved, add a MomentumCorrectionBijector to the list of bijectors
                    correction = MomentumCorrectionBijector(name='MomentumCorrectionBijector_'+str(i))
                    bijectors.extend([bijector, correction, permutation])
                else:
                    # Otherwise, only add the GeneralLorentzTransformBijector to the list of bijectors
                    bijectors.extend([bijector, permutation])
                
        # Calculate shift necessary to restore original order
        restore_shift = self.n_particles - self.n_bijectors % self.n_particles

        # Add a final Permute bijector to restore original order
        if restore_shift != self.n_particles:
            for q in range(restore_shift):
                permutation = tfb.Permute(permutation=base_perm)
                bijector = tfb.Identity()
                bijectors.extend([bijector, permutation])
                
        super(GeneralLorentzNormalizingFlow, self).__init__(list(reversed(bijectors)), validate_args=True, name=name)
        
        self._lorentz_transform_NNs = lorentz_transform_NNs
        self._bijectors_list = bijectors
        
        # If you need to access the bijectors from outside the class, you can define a property for it
        @property
        def bijectors_list(self):
            return self._bijectors_list
        
        # If you need to access the Lorentz transformation selectors from outside the class, you can define a property for it   
        @property   
        def lorentz_transform_NNs(self):
            return self._lorentz_transform_NNs
        

    def call(self, z):
        """
        Transform the input samples using the flow.
        :param z: The input samples to transform.
        :return: The transformed samples.
        """
        return self.forward(z)

def lorentz_transform(particle, beta, theta, phi, theta_1, theta_2, theta_3, dtype="float32"):
    """
    Perform a general Lorentz transformation on a 4-momentum vector.
    The transformation is a boost with rapidity eta along the (phi, theta) direction 
    followed by a rotation around the (theta_1, theta_2, theta_3) axis. 
    The "3-2-3" notation for Euler rotation is used.

    :param particle: tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the input 4-momentum vector.
    :param beta : tf.Tensor
        Tensor of shape (n_events, ), the boost velocity.
    :param theta: tf.Tensor
        Tensor of shape (n_events, ), the polar angle of the boost velocity.
    :param phi: tf.Tensor   
        Tensor of shape (n_events, ), the azimuthal angle of the boost velocity.
    :param theta_1: tf.Tensor 
        Tensor of shape (n_events, ), the first Euler angle.
    :param theta_2: tf.Tensor
        Tensor of shape (n_events, ), the second Euler angle.
    :param theta_3: tf.Tensor
        Tensor of shape (n_events, ), the third Euler angle.
    
    :return: tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the transformed 4-momentum vector.
    """
    # Compute the Lorentz boost matrix and the rotation matrix
    boost_matrix = lorentz_boost(beta, theta, phi, dtype=dtype)  # shape: (n_samples, 4, 4)
    rotation_matrix = euler_rotation(theta_1, theta_2, theta_3, dtype=dtype)  # shape: (n_samples, 4, 4)
    
    # Apply first the boost and then the rotation
    transformed_particle = apply_lorentz_transformation(particle, boost_matrix)
    transformed_particle = apply_lorentz_transformation(transformed_particle, rotation_matrix)
    
    return transformed_particle

def inverse_lorentz_transform(particle, beta, theta, phi, theta_1, theta_2, theta_3, dtype="float32"):
    """
    Perform the inverse Lorentz transformation of lorentz_transform on a 4-momentum vector.
    The "3-2-3" notation for Euler angles is used.

    Parameters:
    same as lorentz_transform

    Returns:
    same as lorentz_transform
    """
    # Compute the inverse Lorentz boost matrix and the inverse rotation matrix
    boost_matrix = lorentz_boost(-beta, theta, phi, dtype=dtype)  # shape: (n_samples, 4, 4)
    rotation_matrix = euler_rotation(-theta_3, -theta_2, -theta_1, dtype=dtype)  # shape: (n_samples, 4, 4)
    
    # Apply first the rotation and then the boost
    transformed_particle = apply_lorentz_transformation(particle, rotation_matrix)
    transformed_particle = apply_lorentz_transformation(transformed_particle, boost_matrix)

    return transformed_particle

def lorentz_boost(beta, theta, phi, dtype="float32"):
    """
    Generates the tensor for a Lorentz boosts on 4-momentum vectors. 
    The boost is parametrized by the speed beta and the angles theta and phi.

    Parameters:
    beta: Tensor of shape (n_samples, n_particles), the speed of the boost for each event and each particle.
    theta: Tensor of shape (n_samples, n_particles), the polar angle of the boost direction for each event and each particle.
    phi: Tensor of shape (n_samples, n_particles), the azimuthal angle of the boost direction for each event and each particle.

    Returns:
    Tensor of shape (n_samples, 4, 4), the boost matrices for each sample.
    """

    # Check input shapes. If inputs are just numbers, convert them to tensors
    if isinstance(beta, (int, float)):
        beta = tf.convert_to_tensor([beta], dtype=dtype)
    if isinstance(theta, (int, float)):
        theta = tf.convert_to_tensor([theta], dtype=dtype)
    if isinstance(phi, (int, float)):
        phi = tf.convert_to_tensor([phi], dtype=dtype)
    
    # Compute the beta components
    beta_x = beta * tf.sin(theta) * tf.cos(phi)
    beta_y = beta * tf.sin(theta) * tf.sin(phi)
    beta_z = beta * tf.cos(theta)
    
    # Compute the number of samples
    n_samples = tf.shape(beta_x)[0]

    # Compute the gamma factor
    gamma = 1 / tf.sqrt(1 - beta**2)  # shape: (n_samples, n_particles)
    # Compute other relevant quantities
    gammam1 = gamma - 1
    
    # Define the zero_beta condition
    zero_beta = tf.less(beta, 1e-10)
    zero = tf.constant(0., dtype=dtype)
    one = tf.constant(1., dtype=dtype)

    # Build the componets of the boost matrix. If beta is zero, the boost matrix is the identity
    b00 = gamma
    b01 = tf.where(zero_beta, zero, -gamma * beta_x)
    b02 = tf.where(zero_beta, zero, -gamma * beta_y)
    b03 = tf.where(zero_beta, zero, -gamma * beta_z)

    b10 = -gamma * beta_x
    b11 = tf.where(zero_beta, one, 1 + gammam1 * beta_x**2 / beta**2)
    b12 = tf.where(zero_beta, zero, gammam1 * beta_x * beta_y / beta**2)
    b13 = tf.where(zero_beta, zero, gammam1 * beta_x * beta_z / beta**2)

    b20 = -gamma * beta_y
    b21 = b12
    b22 = tf.where(zero_beta, one, 1 + gammam1 * beta_y**2 / beta**2)
    b23 = tf.where(zero_beta, zero, gammam1 * beta_y * beta_z / beta**2)

    b30 = b03
    b31 = b13
    b32 = b23
    b33 = tf.where(zero_beta, one, 1 + gammam1 * beta_z**2 / beta**2)

    # Stack the components together and reshape to get a shape of (4, 4, n_samples)
    boost_matrix = tf.reshape(tf.stack([
        b00, b01, b02, b03,
        b10, b11, b12, b13,
        b20, b21, b22, b23,
        b30, b31, b32, b33
    ], axis=0), (4, 4, n_samples))
    # Transpose to get a shape of (n_samples, 4, 4)
    boost_matrix = tf.transpose(boost_matrix, perm=[2, 0, 1])
        
    return boost_matrix

def euler_rotation(theta_1, theta_2, theta_3, dtype="float32"):
    """
    Generates the tensor of Euler rotations on 4-momentum vectors. 
    The rotation angles are specified by the theta_1, theta_2 and theta_3 angles and the "3-2-3" convention is followed.

    Parameters:
    theta_1: Tensor of shape (n_samples, ), the theta_1 angle for the rotation for each particle.
    theta_2: Tensor of shape (n_samples, ), the theta_2 angle for the rotation for each particle.
    theta_3: Tensor of shape (n_samples, ), the theta_3 angle for the rotation for each particle.
    
    Returns:
    Tensor of shape (n_samples, 4, 4), the rotation matrices for each sample.
    """
    # Check input shapes. If inputs are just numbers, convert them to tensors
    if isinstance(theta_1, (int, float)):
        theta_1 = tf.convert_to_tensor([theta_1], dtype=dtype)
    if isinstance(theta_2, (int, float)):
        theta_2 = tf.convert_to_tensor([theta_2], dtype=dtype)
    if isinstance(theta_3, (int, float)):
        theta_3 = tf.convert_to_tensor([theta_3], dtype=dtype)
    
    # Compute the number of samples
    n_samples = tf.shape(theta_1)[0]
    
    # Compute the rotation matrix components (shape: (n_samples, ))
    c1, s1 = tf.math.cos(theta_1), tf.math.sin(theta_1) # shape: (n_samples, )
    c2, s2 = tf.math.cos(theta_2), tf.math.sin(theta_2) # shape: (n_samples, )
    c3, s3 = tf.math.cos(theta_3), tf.math.sin(theta_3) # shape: (n_samples, )
    ones = tf.ones_like(c1) # shape: (n_samples, )
    zeros = tf.zeros_like(c1) # shape: (n_samples, )
    
    #Build the components of the rotation matrix
    r00 = ones
    r01 = zeros
    r02 = zeros
    r03 = zeros
    
    r10 = zeros
    r11 = c1*c2*c3 - s1*s3
    r12 = -c3*s1 - c1*c2*s3
    r13 = c1*s2
    
    r20 = zeros
    r21 = c1*s3 + c2*c3*s1
    r22 = c1*c3 - c2*s1*s3
    r23 = s1*s2
    
    r30 = zeros
    r31 = -c3*s2
    r32 = s2*s3
    r33 = c2
    
    # Stack the rotation matrix components into a tensor of shape (4, 4, n_samples)
    rotation_matrix = tf.reshape(tf.stack([
        r00, r01, r02, r03,
        r10, r11, r12, r13,
        r20, r21, r22, r23,
        r30, r31, r32, r33], axis=0), (4, 4, n_samples))

    # Transpose to get a shape of (n_samples, 4, 4)
    rotation_matrix = tf.transpose(rotation_matrix, perm=[2, 0, 1])
    
    return rotation_matrix # shape: (n_samples, 4, 4)

def apply_lorentz_transformation(particle, transformation_matrix):
    """
    Perform a Lorentz transformation on a 4-momentum vector. The transformation matrix is specified by the transformation_matrix.
    
    Parameters:
    particle: Tensor of shape (n_samples, 4*n_particles), the 4-momentum vector of the particles.
    transformation_matrix: Tensor of shape (n_samples, 4, 4), the Lorentz transformation matrix.
    
    Returns:
    Tensor of shape (n_samples, 4*n_particles), the transformed 4-momentum vector.
    """
    
    # Reshape parameters for consistent dimensions
    n_particles = particle.get_shape()[-1] // 4 # Number of particles in each event
    particle = tf.reshape(particle, [-1, n_particles, 4]) # reshape particle to (n_samples, n_particles, 4)
    
    # Reshape the transformation matrix to match the shape of the particle (n_samples, n_particles, 4, 4)
    transformation_matrix = tf.expand_dims(transformation_matrix, axis=1)
    transformation_matrix = tf.tile(transformation_matrix, [1, n_particles, 1, 1])
    
    # Rotate the momentum components of the particles (shape: (n_samples, n_particles, 3))
    transformed = tf.linalg.matvec(transformation_matrix, particle) # shape: (n_samples, n_particles, 3)
    
    # Flatten the output to match the shape of the input (n_samples, 4*n_particles)
    transformed = tf.reshape(transformed, [-1, n_particles * 4]) # shape: (n_samples, 4*n_particles)
    
    return transformed

def chop_to_zero(number, threshold=1e-06):
    if abs(number) < threshold:
        return 0.0
    else:
        return number
    
def chop_to_zero_array(x, threshold=1e-06):
    if isinstance(x, (list, tuple)):
        return [chop_to_zero(y, threshold) for y in x]
    elif isinstance(x, np.ndarray):
        return np.where(np.abs(x) < threshold, 0.0, x)
    elif isinstance(x, tf.Tensor):
        return tf.where(tf.abs(x) < threshold, 0.0, x)
    else:
        return chop_to_zero(x, threshold)

def check_4momentum_conservation(events):
    """
    Check if energy-momentum conservation is satisfied
    """
    print("The event contains {} particles".format(len(events)))
    print("Checking energy-momentum conservation for each event...")
    result = {}
    
    # Convert events to TensorFlow tensor if it's a NumPy array
    if isinstance(events, np.ndarray):
        events = tf.convert_to_tensor(events)
        
    # Expand dimensions if events is a 1D tensor
    if len(events.shape) < 2:
        events = tf.expand_dims(events, axis=0)
    
    try:
        events = tf.reshape(events.numpy(), (-1, int(events.shape[1] / 4), 4))
    except:
        events = tf.reshape(events, (-1, int(events.shape[1] / 4), 4))
        
    for i in range(len(events)):
        particles = events[i]
        energy = particles[..., 0]
        momentum_x = particles[..., 1]
        momentum_y = particles[..., 2]
        momentum_z = particles[..., 3]
        
        energy_sum = tf.reduce_sum(energy, axis=-1)
        momentum_x_sum = tf.reduce_sum(momentum_x, axis=-1)
        momentum_y_sum = tf.reduce_sum(momentum_y, axis=-1)
        momentum_z_sum = tf.reduce_sum(momentum_z, axis=-1)
        
        result_e = tf.reduce_all(tf.abs(energy_sum) < 1e-6).numpy()
        result_x = tf.reduce_all(tf.abs(momentum_x_sum) < 1e-6).numpy()
        result_y = tf.reduce_all(tf.abs(momentum_y_sum) < 1e-6).numpy()
        result_z = tf.reduce_all(tf.abs(momentum_z_sum) < 1e-6).numpy()
        
        result["event:"+str(i)] = {'energy': result_e, 'momentum_x': result_x, 'momentum_y': result_y, 'momentum_z': result_z}
    
    return result

def compute_masses(events):
    """
    Compute the invariant masses of the particles in the event
    """
    print("The event contains {} particles".format(len(events)))
    print("Computing masses from on-shell condition")
    result = {}
    
    # Convert events to TensorFlow tensor if it's a NumPy array
    if isinstance(events, np.ndarray):
        events = tf.convert_to_tensor(events)
    
    # Expand dimensions if events is a 1D tensor
    if len(events.shape) < 2:
        events = tf.expand_dims(events, axis=0)
    
    try:
        events = tf.reshape(events.numpy(), (-1, int(events.shape[1] / 4), 4))
    except:
        events = tf.reshape(events, (-1, int(events.shape[1] / 4), 4))
        
    for i in range(len(events)):
        particles = events[i]
        masses = tf.sqrt(chop_to_zero_array(particles[..., 0]**2 - particles[..., 1]**2 - particles[..., 2]**2 - particles[..., 3]**2))
        result["event:"+str(i)] = masses.numpy()
    
    return result

def compute_squared_masses(events):
    """
    Compute the squared masses of the particles in the event
    """
    print("The event contains {} particles".format(len(events)))
    print("Computing squared masses from on-shell condition")
    result = {}
    
    # Convert events to TensorFlow tensor if it's a NumPy array
    if isinstance(events, np.ndarray):
        events = tf.convert_to_tensor(events)
    
    # Expand dimensions if events is a 1D tensor
    if len(events.shape) < 2:
        events = tf.expand_dims(events, axis=0)
    
    try:
        events = tf.reshape(events.numpy(), (-1, int(events.shape[1] / 4), 4))
    except:
        events = tf.reshape(events, (-1, int(events.shape[1] / 4), 4))
        
    for i in range(len(events)):
        particles = events[i]
        masses2 = particles[..., 0]**2 - particles[..., 1]**2 - particles[..., 2]**2 - particles[..., 3]**2
        result["event:"+str(i)] = masses2.numpy()
    
    return result

def cornerplotter(target_test_data,nf_dist,max_dim=32,n_bins=50):
    # Define the two samples (target and nf)
    shape = target_test_data.shape
    target_samples=target_test_data
    nf_samples=nf_dist
    ndims = shape[1]
    # Check/remove nans
    nf_samples_no_nans = nf_samples[~np.isnan(nf_samples).any(axis=1), :]
    if len(nf_samples) != len(nf_samples_no_nans):
        print("Samples containing nan have been removed. The fraction of nans over the total samples was:", str((len(nf_samples)-len(nf_samples_no_nans))/len(nf_samples)),".")
    else:
        pass
    nf_samples = nf_samples_no_nans[:shape[0]]
    # Define generic labels
    labels = []
    for i in range(shape[1]):
        labels.append(r"$\theta_{%d}$" % i)
        i = i+1
    # Choose dimensions to plot
    thin = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    # Select samples to plot
    target_samples = target_samples[:,::thin]
    nf_samples = nf_samples[:,::thin]
    # Select labels
    labels = list(np.array(labels)[::thin])

    red_bins=n_bins
    density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins
    #
    blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    blue_bins=blue_bins.astype(int).tolist()

    #file = open(path_to_plots+'/samples.pkl', 'wb')
    #pkl.dump(np.array(target_samples), file, protocol=4)
    #pkl.dump(np.array(nf_samples), file, protocol=4)
    #file.close()

    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner(target_samples,color='red',bins=red_bins,labels=[r"%s" % s for s in labels],normalize1d=True)
    corner(nf_samples,color='blue',bins=blue_bins,fig=figure,normalize1d=True)
    plt.legend(handles=[blue_line,red_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    #plt.savefig(path_to_plots+'/corner_plot.pdf',dpi=300)#pil_kwargs={'quality':50})
    plt.show()
    plt.close()
    return

#def Huber_log_prob_loss(y_true, y_pred, delta=5.0):
#    """Huber loss function for log_prob loss"""
#    # Define threshold
#    delta = tf.cast(delta, y_pred.dtype)
#    # Define error
#    error = -y_pred
#    # Define condition for error
#    condition = tf.abs(error) < delta
#    # Apply quadratic loss for small errors and linear loss for large errors
#    small_error_loss = 0.5 * tf.square(error)
#    large_error_loss = delta * (tf.abs(error) - 0.5 * delta)
#    # Return loss depending on condition
#    return tf.where(condition, small_error_loss, large_error_loss)

class HuberLogProbLoss(tf.keras.losses.Loss):
    def __init__(self, delta=5.0, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        delta = tf.cast(self.delta, y_pred.dtype)
        error = -y_pred
        condition = tf.abs(error) < delta
        small_error_loss = 0.5 * tf.square(error)
        large_error_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(condition, small_error_loss, large_error_loss)


class Trainer_short:
    def __init__(self,
                 callbacks_kwargs=None,
                 ):        
        

#prova = Trainer(data_kwargs={'seed': 0},
#                compiler_kwargs={'optimizer': {'class_name': 'Adam', 'config': {'learning_rate': 0.001}}, 
#                                 'loss': {'class_name': 'HuberLogProbLoss', 'config': {}}},
#                optimizer_kargs={'learning_rate': 0.001},
#                fit_kwargs={'batch_size': 1000, 'epochs': 1000, 'verbose': 0},
#                callbacks_kwargs={'patience': 30, 'min_delta': 0.001, 'reduce_lr_factor': 0.2, 'stop_on_nan': True, 'seed': 0},)

prova = Trainer(data_kwargs={'seed': 0},
                com

class Trainer:
    def __init__(self, 
                 ndims, 
                 trainable_distribution, 
                 X_data, 
                 n_epochs, 
                 batch_size, 
                 n_disp,
                 path_to_results, 
                 load_weights=False, 
                 load_weights_path=None, 
                 lr=.001, 
                 patience=30,
                 min_delta_patience=0.001, 
                 reduce_lr_factor=0.2, 
                 stop_on_nan=True,
                 data_kwargs=None,
                 compiler_kwargs=None,
                 callbacks_kwargs=None,
                 fit_kwargs=None
                 ):

        Utils.reset_random_seeds(data_kwargs.get('seed', 0))
        
        self.X_data = X_data
        self.n_epochs = n_epochs
        self.batch_size = batch_size if batch_size else X_data.shape[0]
        self.path_to_results = path_to_results

        self.x_ = Input(shape=(ndims,), dtype=tf.float32)
        self.log_prob_ = trainable_distribution.log_prob(self.x_)
        self.model = Model(self.x_, self.log_prob_)
        
        # Get compile args
        optimizer_config, loss_config, metrics_configs, compile_kwargs = self._get_compile_args(compiler_kwargs)
        
        # Get optimizer, loss, and metrics from their configs
        optimizer = tf.keras.optimizers.get(optimizer_config)
        loss = self._get_loss(loss_config)
        metrics = [self._get_loss(metric_config) for metric_config in metrics_configs]
        
        # Get callbacks args
        callbacks_configs = self._get_callbacks_args(callbacks_kwargs)
        
        # Get callbacks from their configs
        self._callbacks = self._initialize_callbacks(callbacks_configs)
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **compile_kwargs)
        
        self.load_weights = load_weights

        self.training_time = 0
        self.train_loss=[]
        self.val_loss=[]

        if load_weights:
            self.load_model_weights()
            
    @property
    def optimizer(self):
        return self.model.optimizer

    @property
    def loss(self):
        return self.model.loss

    @property
    def metrics(self):
        return self.model.metrics
    
    @property
    def callbacks(self):
        return self._callbacks
            
    def _get_compile_args(self, compiler_kwargs):
        # Default values
        default_optimizer_config = {'class_name': 'Adam', 'config': {'learning_rate': 0.001}}
        default_loss_config = {'class_name': 'HuberLogProbLoss', 'config': {}}
        default_metrics_configs = [{'class_name': 'log_prob', 'config': {}},
                                   {'class_name': 'HuberLogProbLoss', 'config': {}}]

        # Check if optimizer, loss, and metrics configs are provided
        optimizer_config = compiler_kwargs.get('optimizer', default_optimizer_config)
        loss_config = compiler_kwargs.get('loss', default_loss_config)
        metrics_configs = compiler_kwargs.get('metrics', default_metrics_configs)

        # Get any additional kwargs for model.compile
        compile_kwargs = compiler_kwargs.get('compile_kwargs', {})

        return optimizer_config, loss_config, metrics_configs, compile_kwargs

    def _get_loss(self, loss_config):
        class_name = loss_config['class_name']
        if class_name.lower() == 'huberlogprobloss':
            return HuberLogProbLoss(**loss_config['config'])
        elif class_name.lower() == 'log_prob':
            return lambda _, log_prob: -log_prob
        else:
            raise ValueError(f"Unsupported loss: {class_name}")
        
    def _get_callbacks_args(self, callbacks_kwargs):
        if callbacks_kwargs is None:
            callbacks_kwargs = []

        # Default callbacks configurations
        default_callbacks_configs = [
            {'type': 'LambdaCallback', 'config': {'on_epoch_end': self._epoch_callback}},
            {'type': 'ModelCheckpoint', 'config': {'filepath': self.path_to_results + '/model_checkpoint/weights', 'monitor': 'val_loss', 'save_best_only': True, 'save_weights_only': True}}
        ]

        # Combine provided and default configs
        callbacks_configs = default_callbacks_configs + callbacks_kwargs

        for callback_config in callbacks_configs:
            callback_type = callback_config.get("type", None)
            if callback_type is None:
                raise ValueError("Each callback config dictionary should have a 'type' key.")
        
        return callbacks_configs
    
    def _initialize_callbacks(self, callbacks_configs):
        # Initialize an empty list to hold callbacks
        callbacks = []

        # Loop over the configurations and create callback instances
        for callback_config in callbacks_configs:
            callback_type = callback_config.get("type")
            callback_kwargs = callback_config.get("config", {})

            if callback_type == "LambdaCallback":
                callback = tf.keras.callbacks.LambdaCallback(**callback_kwargs)
            elif callback_type == "ModelCheckpoint":
                callback = tf.keras.callbacks.ModelCheckpoint(**callback_kwargs)
            elif callback_type == "EarlyStopping":
                callback = tf.keras.callbacks.EarlyStopping(**callback_kwargs)
            elif callback_type == "ReduceLROnPlateau":
                callback = tf.keras.callbacks.ReduceLROnPlateau(**callback_kwargs)
            elif callback_type == "TerminateOnNaN":
                callback = tf.keras.callbacks.TerminateOnNaN()
            else:
                raise ValueError(f"Unsupported callback type: {callback_type}")

            callbacks.append(callback)

        return callbacks

    def _epoch_callback(self, epoch, logs):
        n_disp = 1  # or whatever number you want to use
        if epoch % n_disp == 0:
            print('\n Epoch {}/{}'.format(epoch + 1, self.n_epochs, logs),
                  '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))

    def load_model_weights(self):
        # Load weights logic here
        pass

    def fit(self):
        start = timer()
        callbacks = self._initialize_callbacks(callbacks_kwargs)
        history = self.model.fit(x=self.X_data,
                                 y=np.zeros((self.X_data.shape[0], 0), dtype=np.float32),
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs,
                                 validation_split=0.3,
                                 shuffle=True,
                                 verbose=2,
                                 callbacks=callbacks)
        end = timer()
        self.training_time += end - start
    
        history.history['loss']=self.train_loss+history.history['loss']
        history.history['val_loss']=self.val_loss+history.history['val_loss']
        
        return history, self.training_time


def graph_execution(ndims,
                    trainable_distribution, 
                    X_data,
                    n_epochs, 
                    batch_size, 
                    n_disp,
                    path_to_results,
                    load_weights=False,
                    load_weights_path=None,
                    lr=.001,
                    patience=30,
                    min_delta_patience=0.001,
                    reduce_lr_factor=0.2,
                    seed=0,
                    stop_on_nan=True):
    Utils.reset_random_seeds(seed)

    x_ = Input(shape=(ndims,), dtype=tf.float32)
    print(x_)
	
    log_prob_ = trainable_distribution.log_prob(x_)
    print('####### log_prob####')
    print(log_prob_)
    model = Model(x_, log_prob_)

    optimizer = tf.optimizers.Adam(learning_rate=lr,
                                   beta_1=0.9,
                                   beta_2=0.999,
                                   epsilon=1e-07,
                                   amsgrad=True,
                                   weight_decay=None,
                                   clipnorm=1.0,
                                   clipvalue=0.5,
                                   global_clipnorm=None,
                                   use_ema=False,
                                   ema_momentum=0.99,
                                   ema_overwrite_frequency=None,
                                   jit_compile=True,
                                   name='Adam')
    
    #loss = lambda _, log_prob: -log_prob
    loss = Huber_log_prob_loss

    model.compile(optimizer=optimizer,loss=loss)
   
    training_time = 0
    train_loss=[]
    val_loss=[]
    if load_weights==True:
    
        try:
           model.load_weights(path_to_results+'/model_checkpoint/weights')
           print('Found and loaded existing weights.')
           #nf_dist=loader(nf_dist,load_weights_path)      
        except:
            print('No weights found. Training from scratch.')
            
        try:
            with open(path_to_results+'/details.json', 'r') as f:
                # Load JSON data from file
                json_file = json.load(f)
                train_loss = json_file['train_loss_history']
                val_loss = json_file['val_loss_history']
                training_time = json_file['time']
                print('Found and loaded existing history.')
        except:
            print('No history found. Generating new history.')

    ns = X_data.shape[0]
    if batch_size is None:
        batch_size = ns


    #earlystopping
    early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=min_delta_patience, patience=patience*1.2, verbose=1,
    mode='auto', baseline=None, restore_best_weights=True
     )
    #reducelronplateau
    reducelronplateau=tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", min_delta=min_delta_patience, patience=patience, verbose=1,
    factor=reduce_lr_factor, mode="auto", cooldown=0, min_lr=lr/1000
     )
    # Display the loss every n_disp epoch
    epoch_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs:
                        print('\n Epoch {}/{}'.format(epoch+1, n_epochs, logs),
                              '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                       if epoch % n_disp == 0 else False
    )


    checkpoint=tf.keras.callbacks.ModelCheckpoint(
    path_to_results+'/model_checkpoint/weights',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch",
    options=None)
                
    StopOnNAN=tf.keras.callbacks.TerminateOnNaN()

    if stop_on_nan==False:
        callbacks=[epoch_callback,early_stopping,reducelronplateau,checkpoint]
    else:
        callbacks=[epoch_callback,early_stopping,reducelronplateau,checkpoint,StopOnNAN]

    start = timer()
    history = model.fit(x=X_data,
                        y=np.zeros((ns, 0), dtype=np.float32),
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.3,
                        shuffle=True,
                        verbose=2,
                        callbacks=callbacks)
    end = timer()
    training_time = training_time + end - start
    
    history.history['loss']=train_loss+history.history['loss']
    history.history['val_loss']=val_loss+history.history['val_loss']
    
    return history, training_time