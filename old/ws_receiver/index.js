const express = require('express');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
// astronmetry.net config homebrew path:  /opt/homebrew/etc/astrometry.cfg
async function removeAllFiles(folderPath) {
    fs.readdir(folderPath, (err, files) => {
        if (err) {
            console.error('Error reading directory:', err);
            return;
        }
        for (const file of files) {
            const filePath = path.join(folderPath, file);
            fs.unlink(filePath, (err) => {
                if (err) {
                    console.log("failed to delete " + filePath);
                }
            });
        }
    });
}

removeAllFiles("./data"); // Clear data folder (queue) on start

const app = express();
const port = 4242;

const server = app.listen(port, "0.0.0.0", () => {
    console.log(`Server listening on port ${port}`);
});

const wss = new WebSocket.Server({ server });

const client = [];

wss.on('connection', (ws) => {
    console.log('Client connected');
    ws.send(JSON.stringify({status: 'connected'}));
    
    client[ws] = { type: 'none' };
    ws.on('message', (message) => {
        
        try { // Data payload
            const parsed = JSON.parse(message);
            client[ws].type = parsed.fileType;
            console.log(`Set file type to ${parsed.fileType}`);
        } catch {
            // Image sent!
            const base64Data = message.toString('base64');
            const bufferData = Buffer.from(base64Data, 'base64');
            
            const filename = `./data/image_${Date.now()}.${client[ws].type}`;
            fs.writeFile(filename, bufferData, (err) => {
                if (err) {
                    console.error('Error saving image:', err);
                } else {
                    console.log('Image received:', filename);
                    // Convert image to fits
                    const spawn = require("child_process").spawn;
                    const pythonProcess = spawn('python',["./img.py", filename]);
                    pythonProcess.stdout.on('data', (data) => {
                        if (data.toString().includes("Done")) {
                            // Start fits plate solving
                            const fitsFile = filename.substring(0, filename.lastIndexOf(".")) + "-red.fits";
                            //console.log("Conversion successful. Starting plate solve on " + fitsFile + "...");
                            const solveProcess = spawn('solve-field', [fitsFile, "--overwrite", 
                                "--no-plots", 
                                "--sigma", "5.0", 
                                "--radius", "30", 
                                "--no-verify", 
                                "--scale-units", "arcsecperpix", 
                                "-N", "none",
                                "--scale-low", "65", 
                                "--scale-high", "70",
                                "-l", "5"]);
                                
                                
                                solveProcess.stdout.on('data', (data) => {
                                   console.log(data.toString());
                                    if (data.toString().includes("Did not solve")) {
                                        ws.send(JSON.stringify({ solved: false }));
                                        console.log("Solve failed");
                                    } else if (data.toString().includes("Field center: (RA H:M:S, Dec D:M:S)")) { // Store result
                                        let coord = data.toString().substring(data.toString().indexOf("Field center: (RA H:M:S, Dec D:M:S)")).split("=")[1];
                                        let cleaned = coord.trim().slice(0, coord.indexOf(').'));
                                        ws.send(JSON.stringify({ solved: true, coord: cleaned }))
                                        console.log("Solve successful: " + cleaned);
                                    }
                                });
                                
                                // Collect data
                                
                            }
                            
                            
                        });
                    }
                });
            }
            
        });
        
        ws.on('close', () => {
            console.log('Client disconnected');
        });
    });