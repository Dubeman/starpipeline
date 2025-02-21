const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const socket = new WebSocket('ws://localhost:4242');

const testdir = "./testdata";
const testFiles = fs.readdirSync(testdir);

let idx = 0;

socket.onopen = function() {
  socket.send(JSON.stringify({ fileType: "HEIC" }));
  setInterval(() => {
    if (idx < testFiles.length) {
      console.log(`Sending image ${testFiles[idx]}`);
      sendImage(path.join(__dirname, testdir, testFiles[idx]));
      idx++;
    } else {
      clearInterval(this);
    }
  }, 1000 * 5);
};

function sendImage(filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.error('Error reading file', err);
      return;
    }
    socket.send(data);
  });
}

socket.onmessage = function(event) {
  try {
    let data = JSON.parse(event.data);
    if (data.status) {
      console.log("New Status: " + data.status);
    } else {
      if (data.solved) {
        console.log("SOLVED - ", data.coord);
      } else {
        console.log("UNVSOLVED");
      }
    }
  } catch (e) {
    console.log("Could not parse message from server", e);
  }
};

socket.onerror = function(error) {
  console.error('WebSocket Error ', error);  
  
};