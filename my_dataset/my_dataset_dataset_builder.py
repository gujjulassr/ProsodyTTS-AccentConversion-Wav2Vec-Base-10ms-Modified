import tensorflow_datasets as tfds
import torch
import numpy as np
from pathlib import Path  # Import pathlib
import tensorflow as tf

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=("My dataset with Wav2Vec2 features and Mel spectrograms."),
            features=tfds.features.FeaturesDict({
                'features': tfds.features.Tensor(shape=(None, 768), dtype=tf.float32),
                'mels': tfds.features.Tensor(shape=(None,80), dtype=tf.float32),
                'mel_length':tfds.features.Tensor(shape=(),dtype=tf.int32),
                'len_mask':tfds.features.Tensor(shape=(None,1),dtype=tf.float32)
            }),
            supervised_keys=('features', 'mels'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        train = Path('/raid/ai23mtech02001/FastDecTF/train')
        val = Path('/raid/ai23mtech02001/FastDecTF/val')
        test=Path('/raid/ai23mtech02001/FastDecTF/test')

        return {
            'train': self._generate_examples(train / 'features', train / 'mels'),
            'val': self._generate_examples(val / 'features', val / 'mels'),
            'test':self._generate_examples(test / 'features', test / 'mels'),
        }

    def _generate_examples(self, feature_path, mels_path):
        """Yields examples."""
        for feature_file in feature_path.glob('*.pt'):
            mel_file = mels_path / feature_file.name
            if mel_file.exists():
                # Load the .pt files and convert them to NumPy arrays
                features = torch.load(feature_file).detach().numpy()
                mels = torch.load(mel_file).detach().numpy()

                # Transpose mels to have shape (None, 80)
                mels = mels.T  # Transpose to (None, 80)

                mel_length=mels.shape[0]
                len_mask=np.ones((mel_length, 1),dtype=np.float32)


                # Check for None values or invalid data
                if features is None or mels is None or len_mask is None:
                    print(f"Invalid data in {feature_file.stem}")
                    continue

                yield feature_file.stem, {
                    'features': features,
                    'mels': mels,
                    'len_mask':len_mask,
                    'mel_length':mel_length,
                }
    


  