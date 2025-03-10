import React, { useEffect, useRef, useState } from 'react';
import NumpyLoader from '@/utils/numpyLoader';
import { LabelsData, ImageLabel } from '@/types/labelTypes';

interface ImageDisplayProps {
    batchId: number;
    noiseType: string;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ batchId, noiseType }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    const [totalImages, setTotalImages] = useState(0);
    const [images, setImages] = useState<Float32Array[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [labels, setLabels] = useState<LabelsData | null>(null);
    const [showBoxes, setShowBoxes] = useState(true);

    const loadBatch = async () => {
        try {
            setIsLoading(true);
            setError(null);

            console.log('Loading batch:', { noiseType, batchId });
            const response = await fetch(`/api/data/${noiseType}_noise/${noiseType}_images_batch_${batchId}.npy`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            console.log('Response received, reading stream...');
            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('No reader available');
            }

            // Read the stream
            const chunks: Uint8Array[] = [];
            let totalSize = 0;
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                totalSize += value.length;
                console.log('Chunk received:', value.length, 'bytes');
            }

            console.log('Stream complete, total size:', totalSize);

            // Combine chunks
            const arrayBuffer = new ArrayBuffer(totalSize);
            const view = new Uint8Array(arrayBuffer);
            let offset = 0;
            for (const chunk of chunks) {
                view.set(chunk, offset);
                offset += chunk.length;
            }

            console.log('Processing numpy data...');
            const loadedImages = await NumpyLoader.loadNumpyFile(arrayBuffer);
            console.log('Images loaded:', loadedImages.length);

            setImages(loadedImages);
            setTotalImages(loadedImages.length);
            if (loadedImages.length > 0) {
                displayImage(loadedImages[0]);
            }
        } catch (error) {
            console.error('Error loading batch:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
        } finally {
            setIsLoading(false);
        }
    };

    const loadLabels = async () => {
        try {
            console.log('Loading labels for:', { noiseType, batchId });
            const response = await fetch(`/api/data/${noiseType}_noise/${noiseType}_labels.json`);
            if (!response.ok) throw new Error('Failed to load labels');
            const labelsData = await response.json();
            
            // Add detailed logging
            console.log('Total number of labeled images:', labelsData.labels.length);
            console.log('Sample of labels:');
            labelsData.labels.slice(0, 5).forEach((label: ImageLabel, idx: number) => {
                console.log(`Image ${idx}:`, {
                    name: label.image_name,
                    numBoxes: label.bounding_boxes.length
                });
            });

            // Count images with no boxes
            const emptyImages = labelsData.labels.filter((l: ImageLabel) => 
                l.bounding_boxes.length === 0
            ).length;
            console.log('Images with no bounding boxes:', emptyImages);

            setLabels(labelsData);
        } catch (error) {
            console.error('Error loading labels:', error);
        }
    };

    const drawBoundingBoxes = (
        ctx: CanvasRenderingContext2D, 
        imageIndex: number,
        canvasWidth: number,
        canvasHeight: number
    ) => {
        if (!labels || !showBoxes) return;

        // Get base name from Python code format
        const baseImageName = `image_${imageIndex}`;
        
        // Log actual image names for debugging
        console.log('Current image info:', {
            index: imageIndex,
            batchId,
            baseImageName,
            firstFewLabels: labels.labels.slice(0, 3).map(l => l.image_name)
        });

        // Find the label
        const imageLabel = labels.labels.find(l => {
            // Try exact match first
            if (l.image_name === baseImageName) return true;
            
            // Try with batch info
            const batchName = `image_${batchId * 5 + imageIndex}`;
            if (l.image_name === batchName) return true;
            
            return false;
        });

        if (!imageLabel) {
            console.log('No matching label found. Available formats:', {
                tried: [
                    baseImageName,
                    `image_${batchId * 5 + imageIndex}`,
                    `${noiseType}_images_batch_${batchId}_${imageIndex}`
                ]
            });
            return;
        }

        console.log('Found label:', {
            name: imageLabel.image_name,
            numBoxes: imageLabel.bounding_boxes.length,
            boxes: imageLabel.bounding_boxes
        });

        // Draw boxes with more visible style
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 3;
        
        imageLabel.bounding_boxes.forEach(box => {
            // Convert normalized coordinates to canvas coordinates
            const x = box.x_center * canvasWidth;
            const y = box.y_center * canvasHeight;
            const width = box.width * canvasWidth;
            const height = box.height * canvasHeight;

            // Draw box outline
            ctx.strokeStyle = '#FF0000';
            ctx.strokeRect(
                x - width/2,
                y - height/2,
                width,
                height
            );

            // Draw semi-transparent fill
            ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
            ctx.fillRect(
                x - width/2,
                y - height/2,
                width,
                height
            );

            // Draw label background
            const label = `Class ${box.class_id}`;
            ctx.font = 'bold 14px Arial';
            const metrics = ctx.measureText(label);
            const textHeight = 20;
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.fillRect(
                x - width/2,
                y - height/2 - textHeight - 2,
                metrics.width + 10,
                textHeight
            );

            // Draw label text
            ctx.fillStyle = '#FF0000';
            ctx.fillText(
                label,
                x - width/2 + 5,
                y - height/2 - 5
            );
        });
    };

    const displayImage = (imageData: Float32Array) => {
        if (!canvasRef.current) return;
        
        try {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not get canvas context');

            const size = Math.sqrt(imageData.length / 3);
            if (!Number.isInteger(size)) {
                throw new Error(`Invalid image size: ${size}`);
            }

            canvas.width = size;
            canvas.height = size;

            const imageData2D = ctx.createImageData(size, size);
            
            for (let i = 0; i < imageData.length / 3; i++) {
                imageData2D.data[i * 4] = imageData[i * 3] * 255;
                imageData2D.data[i * 4 + 1] = imageData[i * 3 + 1] * 255;
                imageData2D.data[i * 4 + 2] = imageData[i * 3 + 2] * 255;
                imageData2D.data[i * 4 + 3] = 255;
            }

            ctx.putImageData(imageData2D, 0, 0);

            // Draw bounding boxes after image is displayed
            drawBoundingBoxes(ctx, currentImageIndex, size, size);
        } catch (error) {
            console.error('Error displaying image:', error);
            setError(error instanceof Error ? error.message : 'Error displaying image');
        }
    };

    const handleNavigate = (delta: number) => {
        const newIndex = currentImageIndex + delta;
        if (newIndex >= 0 && newIndex < images.length) {
            setCurrentImageIndex(newIndex);
            displayImage(images[newIndex]);
        }
    };

    useEffect(() => {
        loadBatch();
        loadLabels();
    }, [batchId, noiseType]);

    return (
        <div className="flex flex-col items-center min-h-screen w-full 
                        bg-gradient-to-br from-gray-900 via-gray-800 to-indigo-900
                        text-white">
            {/* Header Area */}
            <div className="w-full bg-black/20 backdrop-blur-sm py-4 px-8 mb-8">
                <h2 className="text-xl font-medium text-indigo-200">
                    Deep Space Object Detection
                </h2>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col items-center justify-center gap-8 p-8 w-full
                            bg-gradient-to-b from-transparent via-black/10 to-transparent">
                {/* Image Display */}
                <div className="relative w-[512px] h-[512px] bg-gradient-to-br from-gray-800 to-gray-900 
                               rounded-xl overflow-hidden shadow-2xl ring-1 ring-white/10">
                    {isLoading && (
                        <div className="absolute inset-0 flex items-center justify-center 
                                       bg-black/60 backdrop-blur-sm">
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 bg-indigo-400/80 rounded-full animate-pulse" />
                                <div className="w-3 h-3 bg-indigo-400/80 rounded-full animate-pulse delay-150" />
                                <div className="w-3 h-3 bg-indigo-400/80 rounded-full animate-pulse delay-300" />
                            </div>
                        </div>
                    )}
                    {error && (
                        <div className="absolute inset-0 flex items-center justify-center 
                                       bg-red-500/10 backdrop-blur-sm">
                            <div className="bg-red-50 text-red-700 px-4 py-2 rounded-lg 
                                          shadow-sm border border-red-200">
                                {error}
                            </div>
                        </div>
                    )}
                    <canvas 
                        ref={canvasRef}
                        className="w-full h-full object-contain"
                    />
                </div>

                {/* Controls */}
                <div className="flex items-center gap-6 bg-white/5 backdrop-blur px-6 py-4 
                               rounded-lg shadow-xl ring-1 ring-white/10">
                    <button
                        onClick={() => handleNavigate(-1)}
                        disabled={currentImageIndex === 0}
                        className="px-4 py-2 bg-indigo-500/20 hover:bg-indigo-500/30 
                                  disabled:opacity-40 disabled:hover:bg-indigo-500/20 
                                  rounded-md transition-all duration-150
                                  text-indigo-200 font-medium text-sm 
                                  ring-1 ring-indigo-500/30 hover:shadow-sm 
                                  active:scale-95"
                    >
                        ← Previous
                    </button>
                    
                    <div className="flex items-center gap-3 min-w-[120px] justify-center">
                        <span className="font-medium text-indigo-200">{currentImageIndex + 1}</span>
                        <span className="text-indigo-500">/</span>
                        <span className="text-indigo-300">{totalImages}</span>
                    </div>
                    
                    <button
                        onClick={() => handleNavigate(1)}
                        disabled={currentImageIndex === totalImages - 1}
                        className="px-4 py-2 bg-indigo-500/20 hover:bg-indigo-500/30 
                                  disabled:opacity-40 disabled:hover:bg-indigo-500/20 
                                  rounded-md transition-all duration-150
                                  text-indigo-200 font-medium text-sm 
                                  ring-1 ring-indigo-500/30 hover:shadow-sm 
                                  active:scale-95"
                    >
                        Next →
                    </button>

                    <div className="h-8 w-px bg-indigo-500/20 mx-2" />

                    <label className="flex items-center gap-3 text-sm">
                        <input
                            type="checkbox"
                            checked={showBoxes}
                            onChange={(e) => {
                                setShowBoxes(e.target.checked);
                                if (images[currentImageIndex]) {
                                    displayImage(images[currentImageIndex]);
                                }
                            }}
                            className="w-4 h-4 rounded bg-indigo-500/20 text-indigo-400 
                                      border-indigo-500/30 focus:ring-indigo-500 
                                      transition-colors duration-150"
                        />
                        <span className="text-indigo-200 font-medium">Show Detections</span>
                    </label>
                </div>
            </div>
        </div>
    );
};

export default ImageDisplay;
