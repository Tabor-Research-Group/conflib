
import abc, enum
import numpy as np
from Psience.Molecools import Molecule
import McUtils.Numputils as nput

class BayesianGeneratorModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, initial_data, **opts):
        ...
    @abc.abstractmethod
    def recommend(self) -> dict:
        ...

    @abc.abstractmethod
    def update(self, new_points:dict):
        ...

class BayesianConfigurationGenerator(metaclass=abc.ABCMeta):
    class ConfigurationMode(enum.Enum):
        Cartesian = "cartesian"
        Internal = "internal"
        Dict = "dict"

    def __init__(self,
                 model_type:'type[BayesianGeneratorModel]|BayesianGeneratorModel',
                 mol_spec:Molecule,
                 initial_observations,
                 scoring_function,
                 configuration_mode='cartesians',
                 internals=None,
                 internal_name_mapping=None,
                 **opts
    ):
        self.conf_mode = (
            configuration_mode
                if isinstance(configuration_mode, self.ConfigurationMode) else
            self.ConfigurationMode(configuration_mode.lower())
        )
        self.ref_geom = Molecule.construct(mol_spec)
        if internals is not None:
            self.ref_geom = self.ref_geom.modify(internals=internals)

        self.key_mapping, self.initial_observations = self.prep_initial_observations(
            initial_observations,
            mapping=internal_name_mapping
        )
        self.model = (
            model_type(self.initial_observations, **opts)
                if issubclass(model_type, BayesianGeneratorModel) else
            model_type #TODO: initial_oberservations ignored, should fix in future
        )

        self.scoring_function = scoring_function

    internal_key_fmt='internal-{i}'
    def extract_key_mapping(self, obs:dict):
        return {
            (
                self.internal_key_fmt.format(i=i, k=k)
                    if not (isinstance(k, str) and k == 'values') else
                k
            ): v
            for i, (k, v) in enumerate(obs.items())
        }
    def validate_initial_observations(self, initial_observations:dict):
        if 'values' not in initial_observations:
            key_list = list(initial_observations.keys())
            raise ValueError(
                f"initial observations must contain `values` entry, got {key_list}"
            )
        for k in initial_observations.keys():
            if (
                    not (isinstance(k, str) and k == 'values')
                    and (
                        not isinstance(k, tuple)
                        or not all(nput.is_numeric(ki) for ki in k)
                    )
            ):
                key_list = list(initial_observations.keys())
                raise ValueError(
                    f"initial observations keys must all be internal coordinate positions, got {key_list}"
                )
    def prep_initial_observations(self, initial_observations:dict, mapping=None):
        if mapping is None:
            mapping = self.extract_key_mapping(initial_observations)

        inv_mapping = {v:k for k,v in mapping.items()}
        return mapping, {inv_mapping.get(k, k):v for k,v in initial_observations.items()}

    def prep_sample_points(self, new_sample_points:dict):
        return {
            self.key_mapping(k):np.array(v).reshape(-1)
            for k,v in new_sample_points.items()
        }
    def convert_samples(self, new_sample_points:dict):
        conf_vectors = self.prep_sample_points(new_sample_points)
        if self.conf_mode is self.ConfigurationMode.Dict:
            return conf_vectors
        elif self.conf_mode is self.ConfigurationMode.Internal:
            which = list(new_sample_points.keys())
            vals = np.array(list(new_sample_points.values())).T
            return self.ref_geom.get_displaced_coordinates(
                vals,
                which=which,
                use_internals=True
            )
        elif self.conf_mode is self.ConfigurationMode.Cartesian:
            which = list(new_sample_points.keys())
            vals = np.array(list(new_sample_points.values())).T
            return self.ref_geom.get_displaced_coordinates(
                vals,
                which=which,
                use_internals='reembed'
            )

    def process_next_step(self, samples=None):
        if samples is None:
            samples = self.model.recommend()
        structs = self.convert_samples(samples)
        scores = self.scoring_function(structs)
        samples = dict(
            samples,
            values=scores
        )
        self.model.update(samples)
        return structs, scores
