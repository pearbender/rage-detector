const { app, BrowserWindow } = require('electron/main')
const path = require('node:path')
const dotenv = require('dotenv');

dotenv.config();

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    frame: false,
    transparent: true,
  })

  const url = process.env.URL;

  win.loadURL(url)
}

app.whenReady().then(() => {
  createWindow()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('login', (event, webContents, request, authInfo, callback) => {
  event.preventDefault();
  const username = process.env.USERNAME;
  const password = process.env.PASSWORD;
  callback(username, password);
});