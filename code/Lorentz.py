import json
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
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
    def __init__(self, n_particles, masses=None, means3mom=None, stdev3mom=None, name="BaseDistribution4Momenta", 
                 dtype=tf.float64, reparameterization_type=tfd.FULLY_REPARAMETERIZED, validate_args=False, allow_nan_stats=True, **kwargs):
        """
        Initialize the distribution.
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
        
        # shuffle the particles (3-momenta) in each event
        #samples = tf.transpose(tf.random.shuffle(tf.transpose(samples, [1, 0, 2])), [1, 0, 2])

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
        px = x[..., 1]
        py = x[..., 2]
        pz = x[..., 3]

        # only consider px and py for particles other than the last
        px = px[..., :-1]
        py = py[..., :-1]

        # normal distribution for momenta
        log_prob_px = tf.reduce_sum(tfd.Normal(self.means3mom[..., 0], self.stdev3mom[..., 0]).log_prob(px), axis=-1)
        log_prob_py = tf.reduce_sum(tfd.Normal(self.means3mom[..., 1], self.stdev3mom[..., 1]).log_prob(py), axis=-1)
        log_prob_pz = tf.reduce_sum(tfd.Normal(self.means3mom[..., 2], self.stdev3mom[..., 2]).log_prob(pz), axis=-1)

        return log_prob_px + log_prob_py + log_prob_pz

    def _batch_shape_tensor(self):
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
    def __init__(self, hidden_units, activation='relu', name='LorentzTransformNN', dtype='float32', **kwargs):
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
        self.dense_layers = [tfk.layers.Dense(units, activation=activation, dtype=dtype) for units in hidden_units] # Hidden layers
        self.dense_betas = tfk.layers.Dense(3, activation='sigmoid', dtype=dtype)  # (boost  velocities betas), ranges from -1 to 1
        self.dense_angles = tfk.layers.Dense(3, activation='sigmoid', dtype=dtype)  # (rotation angles theta_i), ranges from 0 to 1 (0 to 2π)

    def call(self, inputs):
        """
        Apply the network to the inputs.
        :param inputs: The inputs. shape: (n_events, n_particles, 4)
        :return: The outputs.
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        betas = 2*self.dense_betas(x)-1  # scale the output of sigmoid to [-1, 1]
        angles = 2 * np.pi * self.dense_angles(x)  # scale the output of sigmoid to [0, 2π]
        parameters = tf.concat([betas, angles], axis=-1)
        return parameters
    
    
class GeneralLorentzTransformBijector(tfb.Bijector):
    """
    A bijector that performs a general Lorentz transformation on a 4-momentum vector.
    """
    def __init__(self, n_particles, lorentz_transform_NN, validate_args=False, name='GeneralLorentzTransformBijector'):
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
        self.n_particles = n_particles
        self.lorentz_transform_NN = lorentz_transform_NN

    def _forward(self, x):
        """
        Perform a general Lorentz transformation on a 4-momentum vector.
        :param x: The 4-momentum vector.
        :return: The transformed 4-momentum vector.
        """
        x_reshaped = tf.reshape(x, [-1, self.n_particles, 4])
        x_particle = x_reshaped[:, 0, :]  # Use the first particle's 4-momentum
        # Use the LorentzTransformNN to choose the Lorentz transformation
        parameters = self.lorentz_transform_NN(x_particle) # shape: (n_samples, 6)
        beta_x = parameters[..., 0]  # Velocity for boost in the x direction (ranges from -1 to 1). shape (n_samples,)
        beta_y = parameters[..., 1]  # Velocity for boost in the y direction (ranges from -1 to 1). shape (n_samples,)
        beta_z = parameters[..., 2]  # Velocity for boost in the z direction (ranges from -1 to 1). shape (n_samples,)
        theta_1 = parameters[..., 3]  # Euler angle 1: rotation around z-axis. shape (n_samples,)
        theta_2 = parameters[..., 4]  # Euler angle 2: rotation around y-axis. shape (n_samples,)
        theta_3 = parameters[..., 5]  # Euler angle 3: rotation around new z-axis. shape (n_samples,)
        # Apply the chosen Lorentz transformation to all particles
        x_transformed = lorentz_transform(x, beta_x, beta_y, beta_z, theta_1, theta_2, theta_3)
        return tf.reshape(x_transformed, tf.shape(x))

    def _inverse(self, y):
        """
        Perform the inverse of a general Lorentz transformation on a 4-momentum vector.
        :param y: The 4-momentum vector.
        :return: The transformed 4-momentum vector.
        """
        y_reshaped = tf.reshape(y, [-1, self.n_particles, 4])
        y_particle = y_reshaped[:, 0, :]  # Use the first particle's 4-momentum
        # Use the LorentzTransformNN to choose the Lorentz transformation
        parameters = self.lorentz_transform_NN(y_particle) # shape: (n_samples, 6)
        beta_x = parameters[..., 0]  # Velocity for boost in the x direction (ranges from -1 to 1). shape (n_samples,)
        beta_y = parameters[..., 1]  # Velocity for boost in the y direction (ranges from -1 to 1). shape (n_samples,)
        beta_z = parameters[..., 2]  # Velocity for boost in the z direction (ranges from -1 to 1). shape (n_samples,)
        theta_1 = parameters[..., 3]  # Euler angle 1: rotation around z-axis. shape (n_samples,)
        theta_2 = parameters[..., 4]  # Euler angle 2: rotation around y-axis. shape (n_samples,)
        theta_3 = parameters[..., 5]  # Euler angle 3: rotation around new z-axis. shape (n_samples,)
        # Apply the inverse of the chosen Lorentz transformation to all particles
        y_transformed = inverse_lorentz_transform(y, beta_x, beta_y, beta_z, theta_1, theta_2, theta_3)
        return tf.reshape(y_transformed, tf.shape(y))
    
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
    def __init__(self, n_particles, n_bijectors, hidden_units, activation='relu', name='ParticleBoostedNormalizingFlow', permute_only=False, **kwargs):
        """
        A normalizing flow that applies a general Lorentz transformation to each particle in an event.
        :param n_particles: The number of particles in the event.
        :param n_bijectors: The number of bijectors in the flow.
        :param hidden_units: The number of hidden units in the Lorentz transformation selector.
        :param activation: The activation function to use in the Lorentz transformation selector.
        :param name: Python `str`, name given to ops managed by this object.
        :param permute_only: Python `bool` indicating whether to only permute the particles. Only useful for debugging paricles permuted by the flow.
        :param kwargs: Additional keyword arguments passed to the `tfk.Model` superclass.
        """
        # If n_bijectors < n_particles, set n_bijectors to n_particles and print a warning
        if n_bijectors < n_particles:
            print("Warning: Number of bijectors was less than the number of particles. Number of bijectors has been set to the number of particles.")
            n_bijectors = n_particles
        
        self.n_particles = n_particles # The number of particles in the event
        self.n_bijectors = n_bijectors # The number of bijectors in the flow
        self.hidden_units = hidden_units  # The number of hidden units in the Lorentz transformation selector
        self.activation = activation # The activation function to use in the Lorentz transformation selector
        bijectors = [] # The list of bijectors that make up the flow
        lorentz_transform_NNs = [] # The list of Lorentz transformation selectors

        # Base permutation order
        base_perm = np.roll(np.arange(4*n_particles), -4)
        
        for i in range(self.n_bijectors):
            # Add a Permute bijector to the list of bijectors
            permutation = tfb.Permute(permutation=base_perm)

            # Always add a LorentzTransformNN bijector
            if permute_only:
                bijector = tfb.Identity()
            else:
                lorentz_transform_NN = LorentzTransformNN(hidden_units=self.hidden_units, activation=self.activation, name='LorentzTransformNN_'+str(i))
                bijector = GeneralLorentzTransformBijector(n_particles=self.n_particles, lorentz_transform_NN=lorentz_transform_NN,  name='GeneralLorentzTransformBijector_'+str(i))
                lorentz_transform_NNs.append(lorentz_transform_NN)
            
            bijectors.extend([bijector, permutation])

            # Cycle the base_perm for next iteration
            # base_perm = np.roll(base_perm, -1)
            
        # Calculate shift necessary to restore original order
        restore_shift = self.n_particles - self.n_bijectors % self.n_particles

        # Add a final Permute bijector to restore original order
        if restore_shift != self.n_particles:
            for q in range(restore_shift):
                permutation = tfb.Permute(permutation=base_perm)
                bijector = tfb.Identity()
                self.bijectors.extend([bijector, permutation])
                
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

def lorentz_transform(particle, beta_x, beta_y, beta_z, theta_1, theta_2, theta_3):
    """
    Perform a general Lorentz transformation on a 4-momentum vector.
    The transformation is a boost with rapidity eta along the (phi, theta) direction 
    followed by a rotation around the (theta_1, theta_2, theta_3) axis. 
    The "3-2-3" notation for Euler rotation is used.

    :param particle: tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the input 4-momentum vector.
    :param beta_x: tf.Tensor
        Tensor of shape (n_events, ), the x-component of the boost velocity.
    :param beta_y: tf.Tensor
        Tensor of shape (n_events, ), the y-component of the boost velocity.
    :param beta_z: tf.Tensor
        Tensor of shape (n_events, ), the z-component of the boost velocity.
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
    boost_matrix = lorentz_boost(beta_x, beta_y, beta_z)  # shape: (n_samples, 4, 4)
    rotation_matrix = euler_rotation(theta_1, theta_2, theta_3)  # shape: (n_samples, 4, 4)
    
    # Apply first the boost and then the rotation
    transformed_particle = apply_lorentz_transformation(particle, boost_matrix)
    transformed_particle = apply_lorentz_transformation(transformed_particle, rotation_matrix)
    
    return transformed_particle

def inverse_lorentz_transform(particle, beta_x, beta_y, beta_z, theta_1, theta_2, theta_3):
    """
    Perform the inverse Lorentz transformation of lorentz_transform on a 4-momentum vector.
    The "3-2-3" notation for Euler angles is used.

    Parameters:
    same as lorentz_transform

    Returns:
    same as lorentz_transform
    """
    # Compute the inverse Lorentz boost matrix and the inverse rotation matrix
    boost_matrix = lorentz_boost(-beta_z, -beta_y, -beta_x)  # shape: (n_samples, 4, 4)
    rotation_matrix = euler_rotation(-theta_3, -theta_2, -theta_1)  # shape: (n_samples, 4, 4)
    
    # Apply first the rotation and then the boost
    transformed_particle = apply_lorentz_transformation(particle, rotation_matrix)
    transformed_particle = apply_lorentz_transformation(transformed_particle, boost_matrix)

    return transformed_particle

def lorentz_boost(beta_x, beta_y, beta_z):
    """
    Generates the tensor for a Lorentz boosts on 4-momentum vectors. 
    The boost is parametrized by the eta_x, eta_y and eta_z rapidities.

    Parameters:
    beta_x: Tensor of shape (n_samples, n_particles), the velocity of the boost along the x axis for each event and each particle.
    beta_y: Tensor of shape (n_samples, n_particles), the velocity of the boost along the x axis for each event and each particle.
    beta_z: Tensor of shape (n_samples, n_particles), the velocity of the boost along the x axis for each event and each particle.

    Returns:
    Tensor of shape (n_samples, 4, 4), the boost matrices for each sample.
    """
    # Check input shapes. If inputs are just numbers, convert them to tensors
    if isinstance(beta_x, (int, float)):
        beta_x = tf.convert_to_tensor([beta_x], dtype=tf.float32)
    if isinstance(beta_y, (int, float)):
        beta_y = tf.convert_to_tensor([beta_y], dtype=tf.float32)
    if isinstance(beta_z, (int, float)):
        beta_z = tf.convert_to_tensor([beta_z], dtype=tf.float32)
    
    # Compute the number of samples
    n_samples = tf.shape(beta_x)[0]

    # Compute the magnitude of the velocity theta_2 and the relativistic factor theta_3
    beta = tf.sqrt(beta_x**2 + beta_y**2 + beta_z**2)  # shape: (n_samples, n_particles)
    gamma = 1 / tf.sqrt(1 - beta**2)  # shape: (n_samples, n_particles)
    # Compute other relevant quantities
    gammam1 = gamma - 1
    
    # Define the zero_beta condition
    zero_beta = tf.less(beta, 1e-10)

    # Build the componets of the boost matrix. If beta is zero, the boost matrix is the identity
    b00 = gamma
    b01 = tf.where(zero_beta, 0., -gamma * beta_x)
    b02 = tf.where(zero_beta, 0., -gamma * beta_y)
    b03 = tf.where(zero_beta, 0., -gamma * beta_z)

    b10 = -gamma * beta_x
    b11 = tf.where(zero_beta, 1., 1 + gammam1 * beta_x**2 / beta**2)
    b12 = tf.where(zero_beta, 0., gammam1 * beta_x * beta_y / beta**2)
    b13 = tf.where(zero_beta, 0., gammam1 * beta_x * beta_z / beta**2)

    b20 = -gamma * beta_y
    b21 = b12
    b22 = tf.where(zero_beta, 1., 1 + gammam1 * beta_y**2 / beta**2)
    b23 = tf.where(zero_beta, 0., gammam1 * beta_y * beta_z / beta**2)

    b30 = b03
    b31 = b13
    b32 = b23
    b33 = tf.where(zero_beta, 1., 1 + gammam1 * beta_z**2 / beta**2)

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

def euler_rotation(theta_1, theta_2, theta_3):
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
        theta_1 = tf.convert_to_tensor([theta_1], dtype=tf.float32)
    if isinstance(theta_2, (int, float)):
        theta_2 = tf.convert_to_tensor([theta_2], dtype=tf.float32)
    if isinstance(theta_3, (int, float)):
        theta_3 = tf.convert_to_tensor([theta_3], dtype=tf.float32)
    
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

#def lorentz_boost(eta, phi, theta):
#    """
#    Generates the tensor for a Lorentz boosts on 4-momentum vectors. 
#    The boost is parametrized by the rapidity eta along the (phi, theta) direction.
#
#    Parameters:
#    eta: Tensor of shape (n_samples, n_particles), the rapidity of the boost for each event and each particle.
#    phi: Tensor of shape (n_samples, n_particles), the phi angle for the boost direction for each event and each particle.
#    theta: Tensor of shape (n_samples, n_particles), the theta angle for the boost direction for each event and each particle.
#
#    Returns:
#    Tensor of shape (n_samples, 4, 4), the boost matrices for each sample.
#    """
#    # Compute the cosh and sinh of the rapidity and the cos and sin of the angles
#    c_eta, s_eta = tf.cosh(eta), tf.sinh(eta)  # shape: (n_samples,)
#    c_theta, s_theta = tf.cos(theta), tf.sin(theta)  # shape: (n_samples,)
#    c_phi, s_phi = tf.cos(phi), tf.sin(phi)  # shape: (n_samples,)
#    
#    # Compute the relevant squares
#    c_theta_sq = c_theta**2
#    s_theta_sq = s_theta**2
#    c_phi_sq = c_phi**2
#    s_phi_sq = s_phi**2
#    
#    # Build the componets of the boost matrix
#    b00 = c_eta
#    b01 = -s_eta * s_theta * c_phi
#    b02 = -s_eta * s_theta * s_phi
#    b03 = -s_eta * c_theta
#    
#    b10 = -s_eta * s_theta * c_phi
#    b11 = 1 + (c_eta - 1) * c_theta_sq * c_phi_sq
#    b12 = (c_eta - 1) * c_theta * s_theta * s_phi
#    b13 = (c_eta - 1) * c_theta * c_theta * s_phi
#    
#    b20 = -s_eta * s_theta * s_phi
#    b21 = (c_eta - 1) * c_theta * s_theta * s_phi
#    b22 = 1 + (c_eta - 1) * s_theta_sq * s_phi_sq
#    b23 = (c_eta - 1) * s_theta * c_theta * s_phi
#    
#    b30 = -s_eta * c_theta
#    b31 = (c_eta - 1) * c_theta * c_theta * s_phi
#    b32 = (c_eta - 1) * s_theta * c_theta * s_phi
#    b33 = 1 + (c_eta - 1) * c_theta_sq
#    
#    # Stack the components together and reshape to get a shape of (4, 4, 5)
#    boost_matrix = tf.reshape(tf.stack([
#        b00, b01, b02, b03,
#        b10, b11, b12, b13,
#        b20, b21, b22, b23,
#        b30, b31, b32, b33
#    ], axis=0), (4, 4, 5))
#    
#    # Transpose to get a shape of (5, 4, 4)
#    boost_matrix = tf.transpose(boost_matrix, perm=[2, 0, 1])
#        
#    return boost_matrix