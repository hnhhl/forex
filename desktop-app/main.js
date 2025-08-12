const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'assets/icon.png'),
    titleBarStyle: 'default',
    backgroundColor: '#0a0a0a'
  });

  // Load the app
  mainWindow.loadFile('src/index.html');

  // Development tools
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App event handlers
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Create application menu
const template = [
  {
    label: 'File',
    submenu: [
      {
        label: 'New Trade',
        accelerator: 'CmdOrCtrl+N',
        click: () => {
          mainWindow.webContents.send('menu-new-trade');
        }
      },
      {
        label: 'Export Data',
        accelerator: 'CmdOrCtrl+E',
        click: () => {
          mainWindow.webContents.send('menu-export');
        }
      },
      { type: 'separator' },
      {
        label: 'Exit',
        accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
        click: () => {
          app.quit();
        }
      }
    ]
  },
  {
    label: 'View',
    submenu: [
      { role: 'reload' },
      { role: 'forcereload' },
      { role: 'toggledevtools' },
      { type: 'separator' },
      { role: 'resetzoom' },
      { role: 'zoomin' },
      { role: 'zoomout' },
      { type: 'separator' },
      { role: 'togglefullscreen' }
    ]
  },
  {
    label: 'Trading',
    submenu: [
      {
        label: 'Quick Buy',
        accelerator: 'CmdOrCtrl+B',
        click: () => {
          mainWindow.webContents.send('quick-buy');
        }
      },
      {
        label: 'Quick Sell', 
        accelerator: 'CmdOrCtrl+S',
        click: () => {
          mainWindow.webContents.send('quick-sell');
        }
      },
      {
        label: 'Close All Positions',
        accelerator: 'CmdOrCtrl+Shift+C',
        click: () => {
          mainWindow.webContents.send('close-all-positions');
        }
      }
    ]
  }
];

const menu = Menu.buildFromTemplate(template);
Menu.setApplicationMenu(menu);