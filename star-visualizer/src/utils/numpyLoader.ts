interface NumpyHeader {
    descr: string;
    fortran_order: boolean;
    shape: number[];
}

export class NumpyLoader {
    private static readonly MAGIC_STRING = '\x93NUMPY';
    private static readonly HEADER_SIZE_OFFSET = 8;
    private static readonly HEADER_SIZE_SIZE = 2;

    /**
     * Loads a batch of images from a .npy file
     * @param arrayBuffer The array buffer containing the .npy file data
     * @returns Array of images as Float32Arrays
     */
    static async loadNumpyFile(arrayBuffer: ArrayBuffer): Promise<Float32Array[]> {
        try {
            // Log the size of received data
            console.log('Received array buffer size:', arrayBuffer.byteLength);

            // Verify minimum size
            if (arrayBuffer.byteLength < 10) {
                throw new Error(`Array buffer too small: ${arrayBuffer.byteLength} bytes`);
            }

            // Check magic string
            const magicBytes = new Uint8Array(arrayBuffer, 0, 6);
            const magicString = String.fromCharCode(...magicBytes);
            console.log('Magic string:', magicString, 'bytes:', [...magicBytes]);

            if (magicString !== this.MAGIC_STRING) {
                throw new Error(`Invalid magic string: ${magicString}`);
            }

            // Get version
            const version = new Uint8Array(arrayBuffer, 6, 2);
            console.log('Version:', [...version]);

            // Get header length
            const headerLength = this.getHeaderDictLength(arrayBuffer);
            console.log('Header length:', headerLength);

            // Calculate total header size
            const totalHeaderSize = this.HEADER_SIZE_OFFSET + this.HEADER_SIZE_SIZE + headerLength;
            console.log('Total header size:', totalHeaderSize);

            if (arrayBuffer.byteLength < totalHeaderSize) {
                throw new Error(`Array buffer smaller than header: ${arrayBuffer.byteLength} < ${totalHeaderSize}`);
            }

            // Get header string
            const headerBytes = new Uint8Array(arrayBuffer, 10, headerLength);
            const headerStr = String.fromCharCode(...headerBytes).trim();
            console.log('Header string:', headerStr);

            // Parse header
            const header = this.parseHeaderDict(headerStr);
            console.log('Parsed header:', header);

            // Validate header
            if (!this.validateHeader(header)) {
                throw new Error(`Invalid header format: ${JSON.stringify(header)}`);
            }

            // Get data
            const dataBuffer = arrayBuffer.slice(totalHeaderSize);
            console.log('Data size:', dataBuffer.byteLength);

            // Verify data size matches shape
            const expectedSize = header.shape.reduce((a, b) => a * b) * 4; // 4 bytes per float32
            if (dataBuffer.byteLength !== expectedSize) {
                throw new Error(`Data size mismatch: got ${dataBuffer.byteLength}, expected ${expectedSize}`);
            }

            const fullArray = new Float32Array(dataBuffer);
            console.log('Array size:', fullArray.length);
            console.log('First few values:', fullArray.slice(0, 5));

            return this.splitIntoImages(fullArray, header.shape);
        } catch (error) {
            console.error('Error in loadNumpyFile:', error);
            throw error;
        }
    }

    /**
     * Parses the numpy header to get array information
     */
    private static parseHeader(arrayBuffer: ArrayBuffer): NumpyHeader {
        try {
            const magicString = String.fromCharCode(...new Uint8Array(arrayBuffer, 0, 6));
            console.log('Magic string:', magicString);
            
            if (magicString !== this.MAGIC_STRING) {
                throw new Error(`Invalid magic string: ${magicString}`);
            }

            const headerLength = this.getHeaderDictLength(arrayBuffer);
            console.log('Header length:', headerLength);
            
            const headerStr = String.fromCharCode(
                ...new Uint8Array(arrayBuffer, 10, headerLength)
            ).trim();
            console.log('Header string:', headerStr);
            
            return this.parseHeaderDict(headerStr);
        } catch (error) {
            console.error('Error parsing header:', error);
            throw error;
        }
    }

    /**
     * Gets the length of the header dictionary from the .npy file
     */
    private static getHeaderDictLength(arrayBuffer: ArrayBuffer): number {
        const view = new DataView(arrayBuffer);
        return view.getUint16(this.HEADER_SIZE_OFFSET, true);
    }

    /**
     * Parses the header dictionary string into an object
     */
    private static parseHeaderDict(headerStr: string): NumpyHeader {
        try {
            console.log('Starting header parsing with:', headerStr);

            // Remove padding spaces and get the actual dictionary part
            const match = headerStr.match(/^\{(.*?)\}/);
            if (!match) {
                throw new Error(`Invalid header format: ${headerStr}`);
            }

            // Get the content between the curly braces
            const dictStr = match[1].trim();
            console.log('Dictionary string after trim:', dictStr);

            const dict: Partial<NumpyHeader> = {};

            // Parse each field individually using regex
            // Parse descr
            const descrMatch = dictStr.match(/'descr':\s*'([^']+)'/);
            if (descrMatch) {
                dict.descr = descrMatch[1];
            }

            // Parse fortran_order
            const fortranMatch = dictStr.match(/'fortran_order':\s*(True|False)/);
            if (fortranMatch) {
                dict.fortran_order = fortranMatch[1] === 'True';
            }

            // Parse shape
            const shapeMatch = dictStr.match(/'shape':\s*\(([\d\s,]+)\)/);
            if (shapeMatch) {
                const shapeStr = shapeMatch[1];
                dict.shape = shapeStr.split(',')
                    .map(s => s.trim())
                    .filter(s => s.length > 0)
                    .map(s => parseInt(s, 10));
                console.log('Parsed shape:', dict.shape);
            }

            // Validate all required fields
            if (!dict.descr) {
                throw new Error('Missing descr field');
            }
            if (dict.fortran_order === undefined) {
                throw new Error('Missing fortran_order field');
            }
            if (!dict.shape || !Array.isArray(dict.shape) || dict.shape.length !== 4) {
                throw new Error(`Invalid shape: ${JSON.stringify(dict.shape)}`);
            }
            if (!dict.shape.every(n => Number.isInteger(n) && n > 0)) {
                throw new Error(`Invalid shape values: ${dict.shape.join(', ')}`);
            }

            console.log('Final parsed dictionary:', dict);
            return dict as NumpyHeader;
        } catch (error) {
            console.error('Header parsing failed:', error);
            throw error;
        }
    }

    /**
     * Validates the header has expected format
     */
    private static validateHeader(header: NumpyHeader): boolean {
        const valid = (
            header.descr === '<f4' &&
            !header.fortran_order &&
            Array.isArray(header.shape) &&
            header.shape.length >= 2
        );

        console.log('Header validation:', {
            hasDescr: !!header.descr,
            descrValue: header.descr,
            hasFortranOrder: header.fortran_order !== undefined,
            fortranValue: header.fortran_order,
            hasShape: Array.isArray(header.shape),
            shapeLength: header.shape?.length,
            isValid: valid
        });

        return valid;
    }

    /**
     * Splits the flat array into individual images based on shape
     */
    private static splitIntoImages(data: Float32Array, shape: number[]): Float32Array[] {
        const [numImages, height, width, channels] = shape;
        const imageSize = height * width * (channels || 1);
        const images: Float32Array[] = [];

        for (let i = 0; i < numImages; i++) {
            const start = i * imageSize;
            const end = start + imageSize;
            images.push(new Float32Array(data.slice(start, end)));
        }

        return images;
    }

    // Add this static method for testing
    static testHeaderParsing(headerStr: string): void {
        console.log('=== Testing Header Parsing ===');
        console.log('Input:', headerStr);
        try {
            const result = this.parseHeaderDict(headerStr);
            console.log('Success! Parsed result:', result);
        } catch (error) {
            console.error('Parsing failed:', error);
        }
        console.log('=== End Test ===');
    }

    static testParsing() {
        const testHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (5, 608, 608, 3), }";
        console.log('Testing header parsing...');
        try {
            const result = this.parseHeaderDict(testHeader);
            console.log('Successfully parsed:', result);
            console.log('Shape:', result.shape);
            console.log('Data type:', result.descr);
            console.log('Fortran order:', result.fortran_order);
        } catch (error) {
            console.error('Test failed:', error);
        }
    }
}

export default NumpyLoader;
