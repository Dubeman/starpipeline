export type NoiseType = 'gaussian' | 'poisson' | 'blur' | 'thermal';

export interface VisualizerState {
    noiseType: NoiseType;
    batchId: number;
    currentImageIndex: number;
    showBoundingBoxes: boolean;
}

export interface VisualizerAction {
    type: 'SET_NOISE_TYPE' | 'SET_BATCH_ID' | 'SET_IMAGE_INDEX' | 'TOGGLE_BOXES';
    payload: any;
} 