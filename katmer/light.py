import jax.numpy as jnp
import pickle
from jax import Array
from jax.typing import ArrayLike

class Light:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Implementing the Singleton Pattern: Ensures only one instance of the Light class exists.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, 
                 wavelength: ArrayLike, 
                 angle_of_incidence: ArrayLike, 
                 polarization: str = 'u'):
        """
        Initialize the Light singleton class with wavelength, angle of incidence, and polarization.

        Parameters:
        - wavelength: jax.numpy.ndarray
            Array of wavelengths.
        - angle_of_incidence: jax.numpy.ndarray
            Array of angles of incidence.
        - polarization: Optional[str]
            Polarization state: 's' for s-polarization, 'p' for p-polarization, or 'u' for unpolarized light.

        Raises:
        - TypeError: If wavelength or angle_of_incidence are not of type jax.numpy.ndarray.
        - ValueError: If polarization is not one of 's', 'p', or 'u'.
        """
        if not isinstance(wavelength, jnp.ndarray):
            raise TypeError("The wavelength array must be of type jax.numpy.ndarray.")
        if not isinstance(angle_of_incidence, jnp.ndarray):
            raise TypeError("The angle_of_incidence array must be of type jax.numpy.ndarray.")
        if polarization not in {'s', 'p', 'u'}:
            raise ValueError("Polarization must be one of 's' (s-polarization), 'p' (p-polarization), or 'u' (unpolarized).")

        self._wavelength = wavelength
        self._wavelength_size = jnp.size(self._wavelength)
        self._angle_of_incidence = angle_of_incidence
        self._angle_of_incidence_size = jnp.size(self._angle_of_incidence)
        if polarization == 's':
            # Unpolarized case: Return tuple (s-polarization, p-polarization)
            self._polarization = jnp.array([0], dtype = jnp.int32)
        elif polarization == 'p':
            # s-polarization case
            self._polarization = jnp.array([1], dtype = jnp.int32)
        elif polarization == 'u':
            # p-polarization case
            self._polarization = jnp.array([2], dtype = jnp.int32)

    @property
    def wavelength(self) -> Array:
        return self._wavelength

    @property
    def angle_of_incidence(self) -> Array:
        return self._angle_of_incidence

    @property
    def polarization(self) -> Array:
        return self._polarization

    @property
    def wavelength_size(self) -> Array:
        return self._wavelength_size

    @property
    def angle_of_incidence_size(self) -> Array:
        return self._angle_of_incidence_size

    def save(self, filename: str):
        """
        Save wavelength, angle of incidence, and polarization data to a pickle file.

        Parameters:
        - filename: str
            Path to the pickle file where the data will be saved.
        """
        data = {
            'wavelength': self._wavelength.tolist(),  # Convert JAX array to Python list for serialization
            'angle_of_incidence': self._angle_of_incidence.tolist(),
            'polarization': self._polarization
        }
        with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(data, file)

    @classmethod
    def load_from_file(cls, filename_without_extension: str) -> 'Light':
        """
        Load Light instance data from a pickle file and return the singleton instance.

        Parameters:
        - filename: str
            Path to the pickle file.

        Returns:
        - Light: Singleton instance with loaded data.
        """
        with open(f"{filename_without_extension}.pkl", "rb") as file:
            data = pickle.load(file)

        return cls(
            wavelength=jnp.array(data['wavelength']),
            angle_of_incidence=jnp.array(data['angle_of_incidence']),
            polarization=data['polarization']
        )