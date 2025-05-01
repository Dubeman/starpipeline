
dataset_path = '/Users/owen/starpipeline/old/starpy/DeepSpaceYoloDataset'
output_path = '/Users/owen/starpipeline/star-visualizer/DeepSpaceYoloDatasetNoisy'

BATCH_SIZE = 1

NOISE_ARGS = {
    'gaussian': {'mean': 0, 'var': 0.01, 'clip': True},
    'poisson': {'clip': True},
    'speckle': {'mean': 0, 'var': 0.01, 'clip': True},
    'blur': {'ksize': (5, 5), 'sigmaX': 0}
}