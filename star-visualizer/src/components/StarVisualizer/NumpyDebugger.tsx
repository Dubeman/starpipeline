import React, { useState } from 'react';

interface NumpyDebuggerProps {
    noiseType: string;
    batchId: number;
}

const NumpyDebugger: React.FC<NumpyDebuggerProps> = ({ noiseType, batchId }) => {
    const [debugInfo, setDebugInfo] = useState<string>('');

    const analyzeNumpyFile = async () => {
        try {
            const response = await fetch(`/api/data/${noiseType}_noise/${noiseType}_images_batch_${batchId}.npy`);
            const buffer = await response.arrayBuffer();
            
            // Create a view of the buffer
            const view = new DataView(buffer);
            const bytes = new Uint8Array(buffer);
            
            let info = '';
            
            // 1. Check magic string (first 6 bytes)
            const magicBytes = bytes.slice(0, 6);
            info += 'Magic String:\n';
            info += `Hex: ${[...magicBytes].map(b => b.toString(16).padStart(2, '0')).join(' ')}\n`;
            info += `ASCII: ${String.fromCharCode(...magicBytes)}\n\n`;
            
            // 2. Version (next 2 bytes)
            const version = bytes.slice(6, 8);
            info += `Version: ${version[0]}.${version[1]}\n\n`;
            
            // 3. Header length (next 2 bytes)
            const headerLength = view.getUint16(8, true);
            info += `Header Length: ${headerLength}\n\n`;
            
            // 4. Header content
            const headerBytes = bytes.slice(10, 10 + headerLength);
            const headerString = String.fromCharCode(...headerBytes);
            info += 'Header Content:\n';
            info += `${headerString}\n\n`;
            
            // 5. Parse header dictionary
            try {
                const headerDict = JSON.parse(
                    headerString
                        .replace(/'/g, '"')  // Replace single quotes with double quotes
                        .replace(/\((.*?)\)/g, '[$1]')  // Replace parentheses with brackets
                );
                info += 'Parsed Header:\n';
                info += JSON.stringify(headerDict, null, 2) + '\n\n';
            } catch (e) {
                info += `Failed to parse header as JSON: ${e}\n\n`;
            }
            
            // 6. Data info
            const dataStart = 10 + headerLength;
            const dataLength = buffer.byteLength - dataStart;
            info += `Data Section:\n`;
            info += `Start: ${dataStart}\n`;
            info += `Length: ${dataLength} bytes\n`;
            info += `First few bytes: ${[...bytes.slice(dataStart, dataStart + 16)]
                .map(b => b.toString(16).padStart(2, '0')).join(' ')}\n`;
            
            setDebugInfo(info);
        } catch (error) {
            setDebugInfo(`Error: ${error}`);
        }
    };

    return (
        <div className="mt-4">
            <button 
                onClick={analyzeNumpyFile}
                className="bg-green-500 text-white px-4 py-2 rounded"
            >
                Analyze Numpy File
            </button>
            {debugInfo && (
                <pre className="mt-2 p-4 bg-gray-100 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                    {debugInfo}
                </pre>
            )}
        </div>
    );
};

export default NumpyDebugger; 