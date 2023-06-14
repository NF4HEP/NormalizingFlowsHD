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
                 dtype=tf.float32, reparameterization_type=tfd.FULLY_REPARAMETERIZED, validate_args=False, allow_nan_stats=True, **kwargs):
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
            masses = tf.zeros((n_particles,))
        self.masses = tf.constant(masses, dtype=tf.float32)

        if means3mom is None:
            means3mom = tf.zeros((3,))
        self.means3mom = tf.constant(means3mom, dtype=tf.float32)

        if stdev3mom is None:
            stdev3mom = tf.ones((3,))
        self.stdev3mom = tf.constant(stdev3mom, dtype=tf.float32)

    def _sample_n(self, n, seed=None):
        """
        Generate n random samples.
        """
        # sample momenta from normal distribution
        samples = tf.random.normal((n, self.n_particles, 3), mean=self.means3mom, stddev=self.stdev3mom)

        # compute momentum sums for conservation
        sum_px = tf.reduce_sum(samples[..., 0], axis=-1, keepdims=True)
        sum_py = tf.reduce_sum(samples[..., 1], axis=-1, keepdims=True)

        # compute remaining px and py for last particle
        samples_fixed_px_py = samples[:, :-1]
        last_px_py = -tf.stack([sum_px, sum_py], axis=-1)[:, 0] + samples[:, -1, :2]
        last_pz = samples[:, -1, 2]

        # reshape last_px_py and last_pz to match dimensions
        last_px_py = tf.reshape(last_px_py, [n, -1])
        last_pz = tf.reshape(last_pz, [n, -1])

        # create new tensor with updated last px, py and pz
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
    def __init__(self, hidden_units, activation='relu', name='LorentzTransformNN', **kwargs):
        """
        Initialize the network.
        :param hidden_units: The number of units in the hidden layers.
        :param activation: The activation function to use in the hidden layers.
        :param name: The name of the network.
        :param kwargs: Additional arguments for the base class.
        """
        super(LorentzTransformNN, self).__init__(name=name, **kwargs)
        self.dense_layers = [tfk.layers.Dense(units, activation=activation) for units in hidden_units] # Hidden layers
        self.dense_phi = tfk.layers.Dense(1, activation='tanh')  # rapidity, ranges from -∞ to +∞
        self.dense_angles = tfk.layers.Dense(5, activation='sigmoid')  # five angles, ranges from 0 to 1 (0 to 2π)

    def call(self, inputs):
        """
        Apply the network to the inputs.
        :param inputs: The inputs. shape: (n_events, n_particles, 4)
        :return: The outputs.
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        rapidity = self.dense_phi(x)
        angles = 2 * np.pi * self.dense_angles(x)  # scale the output of sigmoid to [0, 2π]
        parameters = tf.concat([rapidity, angles], axis=-1)
        return parameters
    

class LorentzAutoregressiveNN(tfk.Model):
    """
    A simple neural network that takes a four-momentum as input and outputs the parameters
    of a Lorentz boost that is applied to the four-momentum.
    """
    def __init__(self, hidden_units, activation='relu', name='LorentzTransformNN', **kwargs):
        """
        Initialize the network.
        :param hidden_units: The number of units in the hidden layers.
        :param activation: The activation function to use in the hidden layers.
        :param name: The name of the network.
        :param kwargs: Additional arguments for the base class.
        """
        super(LorentzTransformNN, self).__init__(name=name, **kwargs)
        self.dense_layers = [tfk.layers.Dense(units, activation=activation) for units in hidden_units] # Hidden layers
        self.dense_phi = tfk.layers.Dense(1, activation='tanh')  # rapidity, ranges from -∞ to +∞
        self.dense_angles = tfk.layers.Dense(5, activation='sigmoid')  # five angles, ranges from 0 to 1 (0 to 2π)

    def call(self, inputs):
        """
        Apply the network to the inputs.
        :param inputs: The inputs. shape: (n_events, n_particles, 4)
        :return: The outputs.
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        rapidity = self.dense_phi(x)
        angles = 2 * np.pi * self.dense_angles(x)  # scale the output of sigmoid to [0, 2π]
        parameters = tf.concat([rapidity, angles], axis=-1)
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
        parameters_expanded = tf.expand_dims(parameters, axis=1) # shape: (n_samples, 1, 6)
        parameters_extended = tf.tile(parameters_expanded, [1, self.n_particles, 1]) # shape: (n_samples, n_particles, 6)
        rapidity = parameters_extended[..., 0]  # Rapidity of the boost direction (ranges from -∞ to +∞). shape (n_events,n_particles)
        phi = parameters_extended[..., 1]  # Polar angle for boost direction (ranges from 0 to π). shape (n_events,n_particles)
        theta = parameters_extended[..., 2]  # Azimuthal angle for boost direction (ranges from 0 to 2π). shape (n_events,n_particles)
        alpha = parameters_extended[..., 3]  # Euler angle 1: rotation around z-axis. shape (n_events,n_particles)
        beta = parameters_extended[..., 4]  # Euler angle 2: rotation around y-axis. shape (n_events,n_particles)
        gamma = parameters_extended[..., 5]  # Euler angle 3: rotation around new z-axis. shape (n_events,n_particles)
        # Apply the chosen Lorentz transformation to all particles
        x_transformed = lorentz_transform(x, rapidity, phi, theta, alpha, beta, gamma)
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
        parameters = self.lorentz_transform_NN(y_particle)
        parameters_expanded = tf.expand_dims(parameters, axis=1)
        parameters_extended = tf.tile(parameters_expanded, [1, self.n_particles, 1])
        rapidity = parameters_extended[..., 0]  # Rapidity of the boost direction (ranges from -∞ to +∞). shape (n_events,n_particles)
        phi = parameters_extended[..., 1]  # Polar angle for boost direction (ranges from 0 to π). shape (n_events,n_particles)
        theta = parameters_extended[..., 2]  # Azimuthal angle for boost direction (ranges from 0 to 2π). shape (n_events,n_particles)
        alpha = parameters_extended[..., 3]  # Euler angle 1: rotation around z-axis. shape (n_events,n_particles)
        beta = parameters_extended[..., 4]  # Euler angle 2: rotation around y-axis. shape (n_events,n_particles)
        gamma = parameters_extended[..., 5]  # Euler angle 3: rotation around new z-axis. shape (n_events,n_particles)
        # Apply the inverse of the chosen Lorentz transformation to all particles
        y_transformed = inverse_lorentz_transform(y, rapidity, phi, theta, alpha, beta, gamma)
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


def lorentz_transform(particle, rapidity, phi, theta, alpha, beta, gamma):
    """
    Perform a general Lorentz transformation on a 4-momentum vector.
    The transformation is a boost with rapidity rapidity along the (phi, theta) direction 
    followed by a rotation around the (alpha, beta, gamma) axis. 
    The ZYZ notation for Euler angles is used.

    :param particle: tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the input 4-momentum vector.
    :param rapidity: tf.Tensor
        Tensor of shape (n_events, n_particles), the rapidity of the boost.
    :param phi: tf.Tensor
        Tensor of shape (n_events, n_particles), the polar angle of the boost direction.
    :param theta: tf.Tensor 
        Tensor of shape (n_events, n_particles), the azimuthal angle of the boost direction.
    :param alpha: tf.Tensor 
        Tensor of shape (n_events, n_particles), the first Euler angle.
    :param beta: tf.Tensor
        Tensor of shape (n_events, n_particles), the second Euler angle.
    :param gamma: tf.Tensor
        Tensor of shape (n_events, n_particles), the third Euler angle.
    
    :return: tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the transformed 4-momentum vector.
    """
    n_particles = particle.get_shape()[-1] // 4 # Number of particles in each event
    # Make a generic boost parametrize by rapidity, phi and theta
    boosted_particle = lorentz_boost(particle, rapidity, phi, theta)  # shape: (n_samples*n_particles, 4)
    # Make a generic rotation parametrized by alpha, beta and gamma
    rotated_particle = euler_rotation(boosted_particle, alpha, beta, gamma)  # shape: (n_samples*n_particles, 4)
    
    return rotated_particle

def inverse_lorentz_transform(particle, rapidity, phi, theta, alpha, beta, gamma):
    """
    Perform a general Lorentz transformation on a 4-momentum vector.
    The ZYZ notation for Euler angles is used.

    Parameters:
    particle: tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the input 4-momentum vector.

    Returns:
    tf.Tensor
        Tensor of shape (n_events, n_particles, 4), the transformed 4-momentum vector.
    """
    n_particles = particle.get_shape()[-1] // 4 # Number of particles in each event
    # Make a generic rotation parametrized by -alpha, -beta and -gamma
    rotated_particle = euler_rotation(particle, -gamma, -beta, -alpha)  # shape: (n_samples*n_particles, 4)
    # Make a generic boost parametrize by -rapidity, phi and theta
    boosted_particle = lorentz_boost(rotated_particle, -rapidity, phi, theta)  # shape: (n_samples*n_particles, 4)
    
    # Reshape the transformed particles back to original shape
    transformed_particle = tf.reshape(boosted_particle, shape=[-1, n_particles, 4])  # shape: (n_samples, n_particles, 4)

    return transformed_particle

def lorentz_boost(particle, rapidity, phi, theta):
    """
    Perform a Lorentz boost on a 4-momentum vector. The boost direction is specified by the rapidity, phi and theta angles. 

    Parameters:
    particle: Tensor of shape (n_samples, 4*n_particles), the 4-momentum vector of the n_particles particles for each event.
    rapidity: Tensor of shape (n_samples, n_particles), the rapidity of the boost for each event and each particle.
    phi: Tensor of shape (n_samples, n_particles), the phi angle for the boost direction for each event and each particle.
    theta: Tensor of shape (n_samples, n_particles), the theta angle for the boost direction for each event and each particle.

    Returns:
    Tensor of shape (n_samples, 4*n_particles), the transformed 4-momentum vector.
    """
    # Reshape parameters for consistent dimensions
    n_particles = particle.get_shape()[-1] // 4 # Number of particles in each event
    particle = tf.reshape(particle, [-1, n_particles, 4]) # reshape particle to (n_samples, n_particles, 4)
    rapidity = tf.expand_dims(rapidity, axis=-1) # reshape rapidity to (n_samples, n_particles, 1)
    phi = tf.expand_dims(phi, axis=-1) # reshape phi to (n_samples, n_particles, 1)
    theta = tf.expand_dims(theta, axis=-1) # reshape theta to (n_samples, n_particles, 1)
    #
    #print("Shape of particle is: ", tf.shape(particle))
    #print("Shape of rapidity is: ", tf.shape(rapidity))
    #print("Shape of phi is: ", tf.shape(phi))
    #print("Shape of theta is: ", tf.shape(theta))
    
    # Compute gamma and beta
    gamma = tf.math.cosh(rapidity)  # shape: (n_samples, n_particles, 1)
    beta = tf.stack([
        tf.math.sinh(rapidity) * tf.math.sin(theta) * tf.math.cos(phi),
        tf.math.sinh(rapidity) * tf.math.sin(theta) * tf.math.sin(phi),
        tf.math.sinh(rapidity) * tf.math.cos(theta)], axis=-1)  # shape: (n_samples, n_particles, 1, 3)
    beta = tf.reshape(beta,[-1,n_particles,3]) # reshape beta to (n_samples, n_particles, 3)
    #
    #print("Shape of gamma is: ", tf.shape(gamma))
    #print("Shape of beta is: ", tf.shape(beta))

    # Extract the energy and momentum components of the particles
    particle_energy = particle[..., 0:1]  # shape: (n_samples, n_particles, 1)
    particle_xyz = particle[..., 1:]  # shape: (n_samples, n_particles, 3)
    #
    #print("Shape of particle_energy is: ", tf.shape(particle_energy))
    #print("Shape of particle_xyz is: ", tf.shape(particle_xyz))

    # Compute the dot product of beta and the particle 3-momentum to get the boost direction component of the energy (dp_beta)
    dp_beta = tf.reduce_sum(particle_xyz * beta, axis=-1, keepdims=True)  # shape: (n_samples, n_particles, 1)
    #
    #print("Shape of dp_beta: ", tf.shape(dp_beta))
    
    # Compute the norm of beta to get the boost magnitude component of the energy (beta_norm)
    beta_norm = tf.norm(beta, axis=-1, keepdims=True) + 1e-10 # shape: (n_samples, n_particles, 1)
    #
    #print("beta_norm shape before division:", tf.shape(beta_norm))

    # Compute the boosted energy and momentum components of the particles (boosted_energy, boosted_xyz)
    boosted_xyz = particle_xyz + (gamma - 1) * dp_beta * beta / beta_norm  # shape: (n_samples, n_particles, 3)
    boosted_energy = gamma * (particle_energy - dp_beta)  # shape: (n_samples, n_particles, 1)
    #
    #print("Shape of boosted_xyz: ", tf.shape(boosted_xyz))
    #print("Shape of boosted_energy: ", tf.shape(boosted_energy))

    # Combine the boosted energy and momentum components of the particles into a single tensor (boosted)
    boosted = tf.concat([boosted_energy, boosted_xyz], axis=-1)  # shape: (n_samples, n_particles, 4)
    #
    #print("Shape of boosted: ", tf.shape(boosted))
    
    # Flatten the output to match the shape of the input (n_samples, 4*n_particles)
    return tf.reshape(boosted, [-1, n_particles * 4])


def euler_rotation(particle, alpha, beta, gamma):
    """
    Perform an Euler rotation on a 4-momentum vector. The rotation angles are specified by the alpha, beta and gamma angles.

    Parameters:
    particle: Tensor of shape (n_samples*n_particles, 4), the 4-momentum vector of particles.
    alpha: Tensor of shape (n_samples*n_particles), the alpha angle for the rotation for each particle.
    beta: Tensor of shape (n_samples*n_particles), the beta angle for the rotation for each particle.
    gamma: Tensor of shape (n_samples*n_particles), the gamma angle for the rotation for each particle.
    
    Returns:
    Tensor of shape (n_samples*n_particles, 4), the rotated 4-momentum vector.
    """
    # Reshape parameters for consistent dimensions
    n_particles = particle.get_shape()[-1] // 4 # Number of particles in each event
    particle = tf.reshape(particle, [-1, n_particles, 4]) # reshape particle to (n_samples, n_particles, 4)
    alpha = tf.expand_dims(alpha, axis=-1) # reshape alpha to (n_samples, n_particles, 1)
    beta = tf.expand_dims(beta, axis=-1) # reshape beta to (n_samples, n_particles, 1)
    gamma = tf.expand_dims(gamma, axis=-1) # reshape gamma to (n_samples, n_particles, 1)
    #
    #print("Shape of particle is: ", tf.shape(particle))
    #print("Shape of alpha is: ", tf.shape(alpha))
    #print("Shape of beta is: ", tf.shape(beta))print("Shape of gamma is: ", tf.shape(gamma))
    
    # Compute the rotation matrix components (shape: (n_samples, n_particles, 1))
    c1, s1 = tf.math.cos(alpha), tf.math.sin(alpha) # shape: (n_samples, n_particles, 1)
    c2, s2 = tf.math.cos(beta), tf.math.sin(beta) # shape: (n_samples, n_particles, 1)
    c3, s3 = tf.math.cos(gamma), tf.math.sin(gamma) # shape: (n_samples, n_particles, 1)
    #
    #print("Shape of c1 and s1 are: ", [tf.shape(c1), tf.shape(s1)])
    #print("Shape of c2 and s2 are: ", [tf.shape(c2), tf.shape(s2)])
    #print("Shape of c3 and s3 are: ", [tf.shape(c3), tf.shape(s3)])

    # Compute the rotation matrix (shape: (n_samples*n_particles, 3, 3))
    rotation_matrix = tf.reshape(tf.concat([
        c2*c3, -c2*s3, s2,
        s1*s2*c3 + c1*s3, -s1*s2*s3 + c1*c3, -s1*c2,
        -c1*s2*c3 + s1*s3, c1*s2*s3 + s1*c3, c1*c2], axis=-1), shape=(-1, n_particles, 3, 3))  # shape: (n_samples*n_particles, 3, 3)
    #
    #print("Shape of rotation_matrix is: ", tf.shape(rotation_matrix))

    # Extract the energy and momentum components of the particles
    particle_energy = particle[..., 0:1]  # shape: (n_samples, n_particles, 1)
    particle_xyz = particle[..., 1:]  # shape: (n_samples, n_particles, 3)
    #
    #print("Shape of particle_energy is: ", tf.shape(particle_energy))
    #print("Shape of particle_xyz is: ", tf.shape(particle_xyz))

    # Rotate the momentum components of the particles (shape: (n_samples, n_particles, 3))
    rotated_xyz = tf.linalg.matvec(rotation_matrix, particle_xyz) # shape: (n_samples, n_particles, 3)
    #
    #print("Shape of rotated_xyz: ", tf.shape(rotated_xyz))

    # Combine the rotated energy and momentum components of the particles into a single tensor (shape: (n_samples, n_particles, 4)) 
    rotated = tf.concat([particle_energy, rotated_xyz], axis=-1)  # shape: (n_samples, n_particles, 4)
    #
    #print("Shape of rotated: ", tf.shape(rotated))

    # Flatten the output to match the shape of the input (n_samples, 4*n_particles)
    return tf.reshape(rotated, [-1, n_particles * 4])


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