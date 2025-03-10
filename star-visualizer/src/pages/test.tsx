import React, { useState } from 'react';

const TestPage: React.FC = () => {
    const [fileContent, setFileContent] = useState<string>('');
    const [error, setError] = useState<string>('');

    const testApi = async () => {
        try {
            setError('');
            const response = await fetch('/api/data/gaussian_noise/gaussian_images_batch_0.npy');
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`HTTP error! status: ${response.status}, message: ${JSON.stringify(errorData)}`);
            }

            const buffer = await response.arrayBuffer();
            const bytes = new Uint8Array(buffer);
            
            // Display first 50 bytes in hex
            const hex = [...bytes].slice(0, 50).map(b => b.toString(16).padStart(2, '0')).join(' ');
            setFileContent(`File size: ${buffer.byteLength} bytes\nFirst 50 bytes: ${hex}`);

        } catch (error) {
            setError(error instanceof Error ? error.message : 'Unknown error');
            console.error('Test failed:', error);
        }
    };

    return (
        <div className="p-4">
            <h1 className="text-2xl mb-4">API Test Page</h1>
            
            <button 
                onClick={testApi}
                className="bg-blue-500 text-white px-4 py-2 rounded"
            >
                Test API
            </button>

            {error && (
                <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
                    {error}
                </div>
            )}

            {fileContent && (
                <pre className="mt-4 p-4 bg-gray-100 rounded overflow-x-auto">
                    {fileContent}
                </pre>
            )}
        </div>
    );
};

export default TestPage; 