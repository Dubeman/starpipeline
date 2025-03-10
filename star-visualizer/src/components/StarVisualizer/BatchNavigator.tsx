import React from 'react';

interface BatchNavigatorProps {
    currentImage: number;
    totalImages: number;
    onNavigate: (index: number) => void;
}

const BatchNavigator: React.FC<BatchNavigatorProps> = ({ 
    currentImage, 
    totalImages, 
    onNavigate 
}) => {
    return (
        <div className="flex items-center gap-4 my-4">
            <button
                onClick={() => onNavigate(currentImage - 1)}
                disabled={currentImage <= 0}
                className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
            >
                Previous
            </button>
            
            <span>
                Image {currentImage + 1} of {totalImages}
            </span>
            
            <button
                onClick={() => onNavigate(currentImage + 1)}
                disabled={currentImage >= totalImages - 1}
                className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
            >
                Next
            </button>
        </div>
    );
};

export default BatchNavigator;
