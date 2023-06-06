import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

class BaseDistribution(tfd.Distribution):
    """
    Base class for a distribution of particles with a fixed number of degrees of freedom.
    The distribution is defined by a set of parameters, which are the parameters of the
    distribution of the individual degrees of freedom. The distribution of the particles
    is obtained by sampling the individual degrees of freedom and then enforcing
    energy-momentum conservation.
    """
    def __init__(self, num_particles, mass_means=None, mass_stds=None, dtype=tf.float32, 
                 validate_args=False, allow_nan_stats=True, name='BaseDistribution'):
        """
        Initialize the distribution.
        :param num_particles: The number of particles in the distribution.
        :param mass_means: Optional list of mean values for the masses of the particles.
        :param mass_stds: Optional list of standard deviations for the masses of the particles.
        :param dtype: The data type of the samples.
        :param validate_args: Whether to validate the arguments.
        :param allow_nan_stats: Whether to allow NaN statistics.
        :param name: The name of the distribution.
        """
        parameters = dict(locals())
        super(BaseDistribution, self).__init__(
            dtype=dtype, reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats,
            parameters=parameters, name=name)
        self.num_particles = num_particles # Number of particles
        self.mass_means = mass_means if mass_means is not None else [0.]*num_particles # Mean values for the masses
        self.mass_stds = mass_stds if mass_stds is not None else [0.01]*num_particles # Standard deviations for the masses

    def _sample_n(self, n, seed=None):
        """
        Sample n particles from the distribution.
        :param n: The number of samples to generate.
        :param seed: The seed for the random number generator.
        :return: The samples.
        """
        # Generate momenta: unit vectors with uniform distribution over the solid angle
        px_py_pz = tfd.Normal(loc=0., scale=1.).sample([n, self.num_particles, 3])
        momenta = tf.nn.l2_normalize(px_py_pz, axis=-1)
        # Enforce three-momentum conservation
        mean_momentum = tf.reduce_mean(momenta, axis=1, keepdims=True)
        momenta -= mean_momentum
        # Generate masses with given means and standard deviations
        masses = tfd.Normal(loc=self.means, scale=self.stds).sample([n, self.num_particles])
        # Set energy equal to magnitude of momentum plus mass term, enforcing on-shell condition
        energies = tf.sqrt(tf.reduce_sum(momenta**2, axis=-1) + masses**2)
        # Combine energies and momenta into a single tensor
        events = tf.concat([energies[..., tf.newaxis], momenta], axis=-1)
        # Reshape tensor to have shape (n, 4*events)
        events = tf.reshape(events, [n, -1])
        return events

    def _log_prob(self, x):
        """
        Compute the log probability of the samples.
        :param x: The samples.
        :return: The log probability of the samples.
        """
        # Use the reciprocal of the square of the energy as the density.
        energy = tf.sqrt(tf.reduce_sum(x[..., 1:]**2, axis=-1))
        return -2.0 * tf.math.log(energy)
    
    
class LorentzTransformSelector(tfk.Model):
    """
    A simple neural network that takes a four-momentum as input and outputs the parameters
    of a Lorentz boost that is applied to the four-momentum.
    """
    def __init__(self, hidden_units, activation='tanh', name='LorentzTransformSelector', **kwargs):
        """
        Initialize the network.
        :param hidden_units: The number of units in the hidden layers.
        :param activation: The activation function to use in the hidden layers.
        :param name: The name of the network.
        :param kwargs: Additional arguments for the base class.
        """
        super(LorentzTransformSelector, self).__init__(name=name, **kwargs)
        self.dense_layers = [tfk.layers.Dense(units, activation=activation) for units in hidden_units] # Hidden layers
        self.dense_phi = tfk.layers.Dense(1, activation='tanh')  # rapidity, ranges from -∞ to +∞
        self.dense_angles = tfk.layers.Dense(5, activation='sigmoid')  # five angles, ranges from 0 to 1 (0 to 2π)

    def call(self, inputs):
        """
        Apply the network to the inputs.
        :param inputs: The inputs.
        :return: The outputs.
        """
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        rapidity = self.dense_phi(x)
        angles = 2 * np.pi * self.dense_angles(x)  # scale the output of sigmoid to [0, 2π]
        parameters = tf.concat([rapidity, angles], axis=-1)
        return parameters
    
    
class GeneralLorentzTransform(tfb.Bijector):
    def __init__(self, phi, theta, psi, alpha, beta, gamma, validate_args=False, name='GeneralLorentzTransform'):
        """
        Initialize the bijector.
        :param phi: Rapidity.
        :param theta: Polar angle for boost direction.
        :param psi: Azimuthal angle for boost direction.
        :param alpha: Euler angle 1: rotation around z-axis.
        :param beta: Euler angle 2: rotation around y-axis.
        :param gamma: Euler angle 3: rotation around new z-axis.
        :param validate_args: Whether to validate the arguments.
        :param name: The name of the bijector.
        """
        self.phi = phi # Rapidity
        self.theta = theta # Polar angle for boost direction
        self.psi = psi # Azimuthal angle for boost direction
        self.alpha = alpha # Euler angle 1: rotation around z-axis
        self.beta = beta # Euler angle 2: rotation around y-axis
        self.gamma = gamma # Euler angle 3: rotation around new z-axis
        super(GeneralLorentzTransform, self).__init__(
            validate_args=validate_args, 
            forward_min_event_ndims=1, 
            name=name)

    def _forward(self, x):
        """
        Apply a Lorentz transformation to the four-momenta.
        """
        x = tf.reshape(x, [-1, self.num_particles, 4])
        return lorentz_transform(x, self.phi, self.theta, self.psi, self.alpha, self.beta, self.gamma)

    def _inverse(self, y):
        """
        Apply the inverse Lorentz transformation to the four-momenta.
        """
        y = tf.reshape(y, [-1, self.num_particles, 4])
        return inverse_lorentz_transform(y, self.phi, self.theta, self.psi, self.alpha, self.beta, self.gamma)

    def _forward_log_det_jacobian(self, x):
        """
        The Jacobian determinant of a Lorentz transformation is 1
        """
        return tf.constant(0.0, dtype=x.dtype)  # The Jacobian determinant of a Lorentz transformation is 1

    def _inverse_log_det_jacobian(self, y):
        """
        The Jacobian determinant of a Lorentz transformation is 1
        """
        return self._forward_log_det_jacobian(y)  # The inverse Jacobian is the same as the forward Jacobian


def lorentz_transform(particle, phi, theta, psi, alpha, beta, gamma):
    """
    Perform a general Lorentz transformation on a 4-momentum vector.
    :param particle: The 4-momentum vector of a particle.
    :param phi: The phi angle for rotation.
    :param theta: The theta angle for rotation.
    :param psi: The psi angle for rotation.
    :param alpha: The x-component of the boost vector.
    :param beta: The y-component of the boost vector.
    :param gamma: The z-component of the boost vector.
    :return: The transformed 4-momentum vector.
    """
    rotation_matrix = rotation_matrix_zyz(phi, theta, psi)
    boost_matrix = boost_matrix(alpha, beta, gamma)
    transformation_matrix = tf.linalg.matmul(rotation_matrix, boost_matrix)
    return tf.linalg.matmul(transformation_matrix, particle)

def inverse_lorentz_transform(particle, phi, theta, psi, alpha, beta, gamma):
    """
    Inverse of lorentz_transform.
    """
    rotation_matrix = rotation_matrix_zyz(-phi, -theta, -psi)
    boost_matrix = boost_matrix(-alpha, -beta, -gamma)
    transformation_matrix = tf.linalg.matmul(boost_matrix, rotation_matrix)
    return tf.linalg.matmul(transformation_matrix, particle)

def rotation_matrix_zyz(alpha, beta, gamma):
    """
    Construct the ZYZ Euler rotation matrix.
    :param alpha: The alpha angle for rotation.
    :param beta: The beta angle for rotation.
    :param gamma: The gamma angle for rotation.
    :return: The rotation matrix.
    """
    # Rotation matrix using Euler angles (Z-Y-Z convention)
    rotation_matrix = tf.linalg.matmul(
        tf.linalg.matmul(
            [[tf.cos(alpha), -tf.sin(alpha), 0, 0],
             [tf.sin(alpha), tf.cos(alpha), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            [[tf.cos(beta), 0, tf.sin(beta), 0],
             [0, 1, 0, 0],
             [-tf.sin(beta), 0, tf.cos(beta), 0],
             [0, 0, 0, 1]]),
        [[tf.cos(gamma), -tf.sin(gamma), 0, 0],
         [tf.sin(gamma), tf.cos(gamma), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])
    return rotation_matrix

def boost_matrix(phi, theta, psi):
    """
    Construct the boost matrix.
    :param phi: The phi angle for boost.
    :param theta: The theta angle for boost.
    :param psi: The psi angle for boost.
    :return: The boost matrix.
    """
    # Boost parameters
    gamma_boost = tf.cosh(phi)
    beta_gamma = tf.sinh(phi)
    # Boost direction
    boost_dir = tf.stack([tf.sin(theta) * tf.cos(psi),
                          tf.sin(theta) * tf.sin(psi),
                          tf.cos(theta)])
    # Boost matrix
    I = tf.eye(4)
    boost_matrix = gamma_boost * I + (gamma_boost - 1) * tf.tensordot(boost_dir, boost_dir, axes=0) - beta_gamma * tf.concat([[[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                                                                                                                              [[[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]],
                                                                                                                              [[[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]]], axis=0)

    return boost_matrix


class GeneralLorentzTransformFlow(tfb.Bijector):
    """
    A bijector that performs a general Lorentz transformation on a 4-momentum vector.
    """
    def __init__(self, num_particles, lorentz_boost_selector, validate_args=False, name='GeneralLorentzTransformFlow'):
        """
        A bijector that performs a general Lorentz transformation on a 4-momentum vector.
        :param num_particles: The number of particles in the event.
        :param lorentz_boost_selector: A function that takes a 4-momentum vector and returns the parameters of the Lorentz transformation.
        :param validate_args: Python `bool` indicating whether arguments should be checked for correctness.
        :param name: Python `str`, name given to ops managed by this object.
        """
        super(GeneralLorentzTransformFlow, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=1, name=name)
        self.num_particles = num_particles
        self.lorentz_boost_selector = lorentz_boost_selector

    def _forward(self, x):
        """
        Perform a general Lorentz transformation on a 4-momentum vector.
        :param x: The 4-momentum vector.
        :return: The transformed 4-momentum vector.
        """
        x_reshaped = tf.reshape(x, [-1, self.num_particles, 4])
        x_particle = x_reshaped[:, 0, :]  # Use the first particle's 4-momentum
        # Use the LorentzTransformSelector to choose the Lorentz transformation
        phi, theta, psi, alpha, beta, gamma = self.lorentz_boost_selector(x_particle)
        # Apply the chosen Lorentz transformation to all particles
        x_transformed = GeneralLorentzTransform(phi, theta, psi, alpha, beta, gamma)._forward(x_reshaped)
        return tf.reshape(x_transformed, tf.shape(x))

    def _inverse(self, y):
        """
        Perform the inverse of a general Lorentz transformation on a 4-momentum vector.
        :param y: The 4-momentum vector.
        :return: The transformed 4-momentum vector.
        """
        y_reshaped = tf.reshape(y, [-1, self.num_particles, 4])
        y_particle = y_reshaped[:, 0, :]  # Use the first particle's 4-momentum
        # Use the LorentzTransformSelector to choose the Lorentz transformation
        phi, theta, psi, alpha, beta, gamma = self.lorentz_boost_selector(y_particle)
        # Apply the inverse of the chosen Lorentz transformation to all particles
        y_transformed = GeneralLorentzTransform(phi, theta, psi, alpha, beta, gamma)._inverse(y_reshaped)
        return tf.reshape(y_transformed, tf.shape(y))

    def _forward_log_det_jacobian(self, x):
        """
        Compute the log det jacobian of the forward transformation.
        :param x: The 4-momentum vector.
        :return: The log det jacobian of the forward transformation.
        """
        # Since the transformation is volume-preserving, the log det jacobian is 0.
        return 0.


class GeneralLorentzNormalizingFlow(tfk.Model):
    """
    A normalizing flow that applies a general Lorentz transformation to each particle in an event.
    """
    def __init__(self, num_particles, hidden_units, activation='tanh', name='ParticleBoostedNormalizingFlow', **kwargs):
        """
        A normalizing flow that applies a general Lorentz transformation to each particle in an event.
        :param num_particles: The number of particles in the event.
        :param hidden_units: The number of hidden units in the Lorentz transformation selector.
        :param activation: The activation function to use in the Lorentz transformation selector.
        :param name: Python `str`, name given to ops managed by this object.
        :param kwargs: Additional keyword arguments passed to the `tfk.Model` superclass.
        """
        super(ParticleBoostedNormalizingFlow, self).__init__(name=name, **kwargs)
        self.num_particles = num_particles # The number of particles in the event
        self.hidden_units = hidden_units  # The number of hidden units in the Lorentz transformation selector
        self.activation = activation # The activation function to use in the Lorentz transformation selector
        self.base_distribution = BaseDistribution(num_particles=num_particles) # The base distribution
        self.bijectors = [] # The list of bijectors that make up the flow
        self.boost_selectors = [] # The list of Lorentz transformation selectors
        for i in range(num_particles):
            boost_selector = LorentzTransformSelector(hidden_units=hidden_units, activation=activation) # The Lorentz transformation selector
            bijector = GeneralLorentzTransformFlow(num_particles=num_particles, lorentz_boost_selector=boost_selector) # The Lorentz transformation
            self.boost_selectors.append(boost_selector) # Add the Lorentz transformation selector to the list
            self.bijectors.append(bijector) # Add the Lorentz transformation to the list
        self.flow = tfb.Chain(list(reversed(self.bijectors))) # We reverse the list of bijectors to construct the flow

    def call(self, n):
        """
        Sample from the flow.
        :param n: The number of samples to draw.
        :return: The samples.
        """
        z = self.base_distribution.sample(n)
        x = self.flow.forward(z)
        return x


def check_4momentum_conservation(events):
    """
    Check if energy-momentum conservation is satisfied
    """
    print("The event contains {} particles".format(len(events)))
    print("Checking energy-momentum conservation for each event...")
    result = {}
    if len(np.shape(events)) < 2:
        events = tf.expand_dims(events, axis=0)
    events = events.numpy().reshape(-1, int(len(events[0])/4), 4)
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
    if len(np.shape(events)) < 2:
        events = tf.expand_dims(events, axis=0)
    events = events.numpy().reshape(-1, int(len(events[0])/4), 4)
    for i in range(len(events)):
        particles = events[i]
        masses = []
        masses = tf.sqrt(particles[..., 0]**2 - particles[..., 1]**2 - particles[..., 2]**2 - particles[..., 3]**2)
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

    #red_bins=50
    #density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins
    #
    #blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    #blue_bins=blue_bins.astype(int).tolist()

    #file = open(path_to_plots+'/samples.pkl', 'wb')
    #pkl.dump(np.array(target_samples), file, protocol=4)
    #pkl.dump(np.array(nf_samples), file, protocol=4)
    #file.close()

    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner.corner(target_samples,color='red',bins=n_bins,labels=[r"%s" % s for s in labels])
    corner.corner(nf_samples,color='blue',bins=n_bins,fig=figure)
    plt.legend(handles=[blue_line,red_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    #plt.savefig(path_to_plots+'/corner_plot.pdf',dpi=300)#pil_kwargs={'quality':50})
    plt.show()
    plt.close()
    return


#class GeneralLorentzTransformFlow(tfb.Bijector):
#    def __init__(self, num_particles, particle_dim):
#        bijectors = []
#        for i in range(num_particles):
#            bijectors.append(LorentzTransformSelector(i, num_particles, particle_dim))
#        super(GeneralLorentzTransformFlow, self).__init__(
#            forward_min_event_ndims=1, 
#            validate_args=False, 
#            name="GeneralLorentzTransformFlow",
#            bijector=tfb.Chain(list(reversed(bijectors))))


#### Sample script ####
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
tfk = tf.keras

num_particles = 4
mass_means = [0.1, 0.2, 0.3, 0.4]
mass_stds = [0.01, 0.02, 0.03, 0.04]
base_dist = BaseDistribution(num_particles=num_particles,
                             mass_means=mass_means,
                             mass_stds=mass_stds)

NN_depth = [128, 128, 128]
num_bijectors = max(num_particles, len(NN_depth))

# ParticleBoostedNormalizingFlow sets up the chain of bijectors for us
flow = ParticleBoostedNormalizingFlow(base_distribution=base_dist, 
                                      num_bijectors=num_bijectors, 
                                      hidden_units=NN_depth)

# Assuming X_data_train and X_data_test are the training and testing data.
X_data_train = ... 
X_data_test = ...

# Loss and optimizer
negloglik = lambda x: -flow.log_prob(x)
optimizer = tfk.optimizers.Adam()

# Training
num_epochs = 10
batch_size = 64

# Define a dataset
dataset = tf.data.Dataset.from_tensor_slices(X_data_train)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size)

for epoch in range(num_epochs):
    print('Start of epoch %d' % (epoch,))
    for step, x_batch_train in enumerate(dataset):
        with tf.GradientTape() as tape:
            loss = negloglik(x_batch_train)
        grads = tape.gradient(loss, flow.trainable_variables)
        optimizer.apply_gradients(zip(grads, flow.trainable_variables))

        if step % 200 == 0:
            print('step %s: mean loss = %s' % (step, tf.reduce_mean(loss)))

# Evaluation
print('Final test set loss: %s' % negloglik(X_data_test))
