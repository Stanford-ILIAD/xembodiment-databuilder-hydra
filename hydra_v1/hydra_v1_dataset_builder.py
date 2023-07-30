from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

#from attrdict import AttrDict

class HydraV1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """
        Dataset metadata 
        real robot dataset for project HYDRA: Hybrid Robot Actions for Imitation Learning
        homepage: https://sites.google.com/view/hydra-il-2023
        robot: Franka Emika Panda
        tasks: make coffee, make toast
        raw npz file states:
            ee_position (663, 3)
            ee_orientation (663, 4)
            ee_orientation_eul (663, 3)
            q (663, 7)
            qdot (663, 7)
            gripper_width (663, 1)
            gripper_pos (663, 1)
            gripper_open (663, 1)
            image (663, 240, 320, 3)
            ego_image (663, 240, 320, 3)
            action (663, 7)
            click_state (663, 1)
            reward (663, 1)
            done (663,)
            rollout_timestep (663,)
        """
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(27,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x EEF position,'
                                '4x EEF orientation in quaternion,'
                                '3x EEF orientation in euler angle,'
                                '7x robot joint angles, '
                                '7x robot joint velocities,' 
                                '3x gripper state.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x EEF positional delta, '
                            '3x EEF orientation delta in euler angle, 1x close gripper].',
                    ),
                    'is_dense': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if state is a waypoint(010) or in dense mode(x111).'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            data = dict(np.load(episode_path, allow_pickle=True))
            print(episode_path)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            langauge = ""
            if "coffee" in episode_path:
                language = "make a cup of coffee with the keurig machine"
            elif "toast" in episode_path:
                language = "make a piece of toast with the oven"
            elif "dishes" in episode_path:
                language = "palce dishes in the dish rack"
            else:
                language = "do something"

            for i in range(data['rollout_timestep'].shape[0]):
                # compute Kona language embedding
                language_embedding = self._embed([language])[0].numpy()
                robot_state = np.concatenate([data['ee_position'][i],data['ee_orientation'][i],
                    data['ee_orientation_eul'][i],data['q'][i],data['qdot'][i],
                    data['gripper_width'][i],data['gripper_pos'][i],data['gripper_open'][i]])

                episode.append({
                    'observation': {
                        'image': data['image'][i],
                        'wrist_image': data['ego_image'][i],
                        'state': robot_state,
                    },
                    'action': data['action'][i],
                    'is_dense': data['click_state'][i][0],
                    'discount': 1.0,
                    'reward': float(i == ( data['rollout_timestep'].shape[0]- 1)),
                    'is_first': i == 0,
                    'is_last': i == (data['rollout_timestep'].shape[0] - 1),
                    'is_terminal': i == (data['rollout_timestep'].shape[0] - 1),
                    'language_instruction': language,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            del data
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob('data/train/*.npz')

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        #beam = tfds.core.lazy_imports.apache_beam
        #return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        #)

