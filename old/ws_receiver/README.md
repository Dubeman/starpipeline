### Websocket Simple Star Solve

Opens a local websocket on port 4242 that will accept any image format, 
convert it to .fits, and solve it with a local astrometry.net installation.

The socket expects that the first message to be JSON and include a `fileType` 
property indictating the format of the images it will be sending. 

Prerequisites: 
- Follow instructions for local installation of astrometry.net. https://astrometry.net/use.html
- Install Node (with NPM) and Python (with PIP)
- `npm install`
- `pip install -r requirements.txt`

To run: `node index.js`
To test `test_client.js`

