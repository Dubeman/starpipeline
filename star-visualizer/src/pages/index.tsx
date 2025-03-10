import React, { useState } from 'react';
import ImageDisplay from '@/components/StarVisualizer/ImageDisplay';

type NoiseType = 'gaussian' | 'poisson' | 'blur' | 'thermal';

const Home: React.FC = () => {
    const [batchId, setBatchId] = useState(0);
    const [noiseType, setNoiseType] = useState<NoiseType>('gaussian');

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
                    <h1 className="text-3xl font-bold text-gray-900">
                        Deep Space Object Visualizer
                    </h1>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
                {/* Controls Section */}
                <div className="bg-white rounded-lg shadow p-6 mb-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Noise Type Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Noise Type
                            </label>
                            <select 
                                value={noiseType} 
                                onChange={(e) => setNoiseType(e.target.value as NoiseType)}
                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 rounded-md"
                            >
                                <option value="gaussian">Gaussian Noise</option>
                                <option value="poisson">Poisson Noise</option>
                                <option value="blur">Blur Effect</option>
                                <option value="thermal">Thermal Noise</option>
                            </select>
                        </div>

                        {/* Batch Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Batch Number
                            </label>
                            <div className="mt-1 flex rounded-md shadow-sm">
                                <input 
                                    type="number" 
                                    value={batchId}
                                    onChange={(e) => setBatchId(parseInt(e.target.value) || 0)}
                                    min={0}
                                    className="flex-1 min-w-0 block w-full px-3 py-2 rounded-md border-gray-300 focus:ring-indigo-500 focus:border-indigo-500"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Visualization Section */}
                <div className="bg-white rounded-lg shadow">
                    <ImageDisplay batchId={batchId} noiseType={noiseType} />
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-white border-t border-gray-200 mt-8">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                    <p className="text-center text-gray-500 text-sm">
                        Deep Space Object Detection Visualization Tool
                    </p>
                </div>
            </footer>
        </div>
    );
};

export default Home;
