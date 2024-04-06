const { app, BrowserWindow } = require('electron/main')
const path = require('node:path')
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '.env') });

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    frame: false,
    transparent: true
  })

  const url = process.env.RAGE_DETECTOR_URL;

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
  const username = process.env.RAGE_DETECTOR_USERNAME;
  const password = process.env.RAGE_DETECTOR_PASSWORD;
  callback(username, password);
});