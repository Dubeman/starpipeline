import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { VisualizerState, VisualizerAction, NoiseType } from '@/types/noiseTypes';

const initialState: VisualizerState = {
    noiseType: 'gaussian',
    batchId: 0,
    currentImageIndex: 0,
    showBoundingBoxes: true
};

function visualizerReducer(state: VisualizerState, action: VisualizerAction): VisualizerState {
    switch (action.type) {
        case 'SET_NOISE_TYPE':
            return { ...state, noiseType: action.payload as NoiseType };
        case 'SET_BATCH_ID':
            return { ...state, batchId: action.payload };
        case 'SET_IMAGE_INDEX':
            return { ...state, currentImageIndex: action.payload };
        case 'TOGGLE_BOXES':
            return { ...state, showBoundingBoxes: action.payload };
        default:
            return state;
    }
}

const VisualizerContext = createContext<{
    state: VisualizerState;
    dispatch: React.Dispatch<VisualizerAction>;
} | undefined>(undefined);

export function VisualizerProvider({ children }: { children: ReactNode }) {
    const [state, dispatch] = useReducer(visualizerReducer, initialState);

    return (
        <VisualizerContext.Provider value={{ state, dispatch }}>
            {children}
        </VisualizerContext.Provider>
    );
}

export function useVisualizer() {
    const context = useContext(VisualizerContext);
    if (context === undefined) {
        throw new Error('useVisualizer must be used within a VisualizerProvider');
    }
    return context;
} 