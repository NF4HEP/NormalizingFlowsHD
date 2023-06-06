import numpy as np
import tensorflow as tf
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
        masses = [tfd.Normal(loc=mean, scale=std).sample([n]) for mean, std in zip(self.mass_means, self.mass_stds)]
        masses = tf.stack(masses, axis=-1)  # Shape: [n, num_particles]

        # Set energy equal to magnitude of momentum plus mass term, enforcing on-shell condition
        energies = tf.sqrt(tf.reduce_sum(momenta**2, axis=-1) + masses**2)

        particles = tf.concat([energies[..., tf.newaxis], momenta], axis=-1)

        # Reshape tensor to have shape (n, 4*nparticles)
        particles = tf.reshape(particles, [n, -1])

        return particles

    def _log_prob(self, x):
        """
        Compute the log probability of the samples.
        :param x: The samples.
        :return: The log probability of the samples.
        """
        # Use the reciprocal of the square of the energy as the density.
        energy = tf.sqrt(tf.reduce_sum(x[..., 1:]**2, axis=-1))
        return -2.0 * tf.math.log(energy)
    
    
    
class GeneralLorentzBoost(tfb.Bijector):
    def __init__(self, phi, theta, psi, alpha, beta, gamma, validate_args=False, name='general_lorentz_boost'):
        """
        Let's define phi as the rapidity, the magnitude of the boost, 
        theta and psi as the polar and azimuthal angles defining the direction of the boost, 
        and alpha, beta, gamma as the Euler angles defining the rotation.
        """
        self.phi = phi # Rapidity
        self.theta = theta # Polar angle for boost direction
        self.psi = psi # Azimuthal angle for boost direction
        self.alpha = alpha # Euler angle 1: rotation around z-axis
        self.beta = beta # Euler angle 2: rotation around y-axis
        self.gamma = gamma # Euler angle 3: rotation around new z-axis
        super(GeneralLorentzBoost, self).__init__(
            forward_min_event_ndims=2,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        """
        Performs a general Lorentz transformation on a four-momentum
        """
        # Boost parameters
        gamma_boost = tf.cosh(self.phi)
        beta_gamma = tf.sinh(self.phi)

        # Boost direction
        boost_dir = tf.stack([tf.sin(self.theta) * tf.cos(self.psi),
                              tf.sin(self.theta) * tf.sin(self.psi),
                              tf.cos(self.theta)])

        # Boost matrix
        I = tf.eye(4)
        boost_matrix = gamma_boost * I + (gamma_boost - 1) * tf.tensordot(boost_dir, boost_dir, axes=0) - beta_gamma * tf.concat([[[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                                                                                                                                      [[[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]],
                                                                                                                                      [[[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]]], axis=0)

        # Rotation matrix using Euler angles (Z-Y-Z convention)
        rotation_matrix = tf.linalg.matmul(
            tf.linalg.matmul(
                [[tf.cos(self.alpha), -tf.sin(self.alpha), 0, 0],
                 [tf.sin(self.alpha), tf.cos(self.alpha), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],
                [[tf.cos(self.beta), 0, tf.sin(self.beta), 0],
                 [0, 1, 0, 0],
                 [-tf.sin(self.beta), 0, tf.cos(self.beta), 0],
                 [0, 0, 0, 1]]),
            [[tf.cos(self.gamma), -tf.sin(self.gamma), 0, 0],
             [tf.sin(self.gamma), tf.cos(self.gamma), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

        # Combine transformations
        lorentz_matrix = tf.matmul(rotation_matrix, boost_matrix)

        return tf.einsum('ij,bkj->bki', lorentz_matrix, x)

    def _inverse(self, y):
        """
        The inverse of a Lorentz transformation is just another Lorentz transformation with the same parameters
        """
        return self._forward(y)  # The inverse of a Lorentz transformation is just another Lorentz transformation

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


def check_4momentum_conservation(events):
    """
    Check if energy-momentum conservation is satisfied
    """
    print("The event contains {} particles".format(len(events)))
    print("Checking energy-momentum conservation for each event...")
    result = {}
    if len(np.shape(events)) < 2:
        events = tf.expand_dims(events, axis=0)
    events = events.numpy().reshape(-1, int(len(samples[0])/4), 4)
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
    events = events.numpy().reshape(-1, int(len(samples[0])/4), 4)
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