import { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';
import { createReadStream, statSync } from 'fs';
import getConfig from 'next/config';

const { serverRuntimeConfig } = getConfig();

export const config = {
    api: {
        bodyParser: false,
        responseLimit: false,
    },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
    try {
        const { path: urlPath } = req.query;
        
        if (!urlPath || !Array.isArray(urlPath)) {
            throw new Error(`Invalid path: ${JSON.stringify(urlPath)}`);
        }

        // Log the requested path components
        console.log('URL path components:', urlPath);

        // Sanitize the path components
        const safePath = urlPath.map(segment => 
            segment.replace(/[^a-zA-Z0-9_\-\.]/g, '')
        );
        
        // Construct the file path
        const filePath = path.join(
            process.cwd(), 
            'DeepSpaceYoloDatasetNoisy',
            ...safePath
        );

        // Log the full resolved path
        console.log('Full file path:', filePath);

        // Check if file exists
        if (!fs.existsSync(filePath)) {
            console.error('File not found:', filePath);
            return res.status(404).json({ error: 'File not found' });
        }

        // Read and check file stats
        const stats = statSync(filePath);
        console.log('File stats:', {
            size: stats.size,
            isFile: stats.isFile(),
            created: stats.birthtime,
            modified: stats.mtime
        });

        // Set headers for streaming
        res.writeHead(200, {
            'Content-Type': 'application/octet-stream',
            'Content-Length': stats.size,
            'Content-Disposition': `attachment; filename=${path.basename(filePath)}`,
        });

        // Create read stream and pipe to response
        const stream = createReadStream(filePath);
        stream.pipe(res);

        // Handle stream errors
        stream.on('error', (error) => {
            console.error('Stream error:', error);
            res.end();
        });

    } catch (error) {
        console.error('Error serving file:', error);
        res.status(500).json({ 
            error: 'Server error',
            message: error instanceof Error ? error.message : 'Unknown error',
            path: req.query.path
        });
    }
}
