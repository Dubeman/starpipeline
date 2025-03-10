import React, { useState } from 'react';

interface FileDebuggerProps {
    noiseType: string;
    batchId: number;
}

const FileDebugger: React.FC<FileDebuggerProps> = ({ noiseType, batchId }) => {
    const [debug, setDebug] = useState<string>('');

    const examineFile = async () => {
        try {
            const response = await fetch(`/api/data/${noiseType}_noise/${noiseType}_images_batch_${batchId}.npy`);
            const buffer = await response.arrayBuffer();
            
            // Examine the first 256 bytes to ensure we capture the full header
            const bytes = new Uint8Array(buffer, 0, 256);
            
            let debugInfo = `File size: ${buffer.byteLength} bytes\n\n`;
            
            // Check magic string
            const magicString = String.fromCharCode(...bytes.slice(0, 6));
            debugInfo += `Magic string: ${magicString}\n`;
            
            // Check version
            const version = [...bytes.slice(6, 8)].map(b => b.toString()).join('.');
            debugInfo += `Version: ${version}\n`;
            
            // Get header length
            const headerLength = new DataView(buffer).getUint16(8, true);
            debugInfo += `Header length: ${headerLength}\n\n`;
            
            // Get header string and parse it
            const headerStr = String.fromCharCode(...bytes.slice(10, 10 + headerLength));
            debugInfo += `Raw header string: ${headerStr}\n`;
            
            // Try to parse the shape
            try {
                const match = headerStr.match(/'shape':\s*\((.*?)\)/);
                if (match) {
                    debugInfo += `Shape string: ${match[1]}\n`;
                    const numbers = match[1].split(',')
                        .map(s => s.trim())
                        .filter(s => s)
                        .map(s => parseInt(s));
                    debugInfo += `Parsed shape: [${numbers.join(', ')}]\n`;
                } else {
                    debugInfo += 'No shape found in header\n';
                }
            } catch (e) {
                debugInfo += `Error parsing shape: ${e}\n`;
            }
            
            debugInfo += '\nFirst 256 bytes:\n';
            debugInfo += 'Hex: ' + [...bytes].map(b => b.toString(16).padStart(2, '0')).join(' ') + '\n';
            debugInfo += 'ASCII: ' + [...bytes].map(b => b >= 32 && b <= 126 ? String.fromCharCode(b) : '.').join('');
            
            setDebug(debugInfo);
        } catch (error) {
            setDebug(`Error: ${error}`);
        }
    };

    return (
        <div className="mt-4">
            <button 
                onClick={examineFile}
                className="bg-blue-500 text-white px-4 py-2 rounded"
            >
                Debug File
            </button>
            {debug && (
                <pre className="mt-2 p-4 bg-gray-100 rounded text-sm overflow-x-auto">
                    {debug}
                </pre>
            )}
        </div>
    );
};

export default FileDebugger; 