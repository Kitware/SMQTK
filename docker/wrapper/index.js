
const express = require('express');
const request = require('request');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ctMap = require('./content-type-map');

const createLink = (path, callback) => {
  let hash = crypto.createHash('sha1');
  let stream = fs.createReadStream(path).pipe(hash);
  let buffers = [];
  stream.on('data', (buffer) => { buffers.push(buffer); });
  stream.on('end', () => {
    let hexString = Buffer.concat(buffers).toString('hex');
    let linkPath = '/links/' + hexString;
    if (!fs.existsSync(linkPath)) {
      fs.symlink('..' + path, linkPath, () => {
        callback(true);
      });
    } else {
      callback(false);
    }
  });
};

const createInitialLinks = (callback) => {
  fs.readdir('/data', (err, files) => {
    let numProcessed = 0;
    const processed = () => {
      ++numProcessed;
      if (numProcessed === files.length) {
        callback();
        callback = () => {};
      }
    };

    files.forEach((file) => {
      let filePath = '/data/' + file;
      if (file[0] === '.') {
        processed();
      } else {
        fs.stat(filePath, (err, stats) => {
          if (stats.isFile()) {
            createLink(filePath, processed);
          } else {
            processed();
          }
        });
      }
    });
  });
};

const createNewLink = (req, callback) => {
  let name = Date.now().toString(36);
  let path = '/newdata/' + name;

  req.on('response', (res) => {
    let type = res.headers['content-type'];
    let ext = ctMap.typeToExtension(type);
    if (ext) {
      path += '.' + ext;
    }

    let stream = req.pipe(fs.createWriteStream(path));
    stream.on('finish', () => {
      createLink(path, (created) => {
        if (created) {
          callback();
        } else {
          fs.unlink(path, () => { callback(); });
        }
      });
    });
  });
};

const proxy = (options) => {
  let result = express();

  result.route('*').all((req, res, next) => {
    let path = req.originalUrl;
    if (path[0] === '/') {
      path = path.slice(1);
    }

    let { protocol, host, port } = options;

    let proxyReq = request(protocol + '://' + host + ':' + port + '/' + path);
    req.pipe(proxyReq).pipe(res);
  });

  return result;
};

const app0 = express();
const app1 = express();

const tripPred = (x) => (
  x !== (void 0) &&
  x !== null &&
  isNaN(Number.parseInt(x))
);

const interceptPath = (req, res, next) => {
  let { n, start, end, path } = req.params;
  let uri = [];
  let tripped = tripPred(n);
  if (tripped) { uri.push(n); } else { tripped = tripPred(start); }
  if (tripped) { uri.push(start); } else { tripped = tripPred(end); }
  if (tripped) { uri.push(end); }
  uri.push(path);

  uri = uri.join('/');

  if (!uri.match(/^[a-z]+:\/\//)) {
    uri = 'http://' + uri;
  }

  createNewLink(request(uri), () => { next(); });
};

app0.get('/image/:hash', (req, res) => {
  let { hash } = req.params;
  let linkPath = '/links/' + hash;
  if (!fs.existsSync(linkPath)) {
    res.sendStatus(404);
  } else {
    fs.readlink(linkPath, (err, realPath) => {
      let resolvedPath = path.resolve(path.join(
        path.dirname(linkPath),
        realPath
      ));

      res.sendFile(resolvedPath);
    });
  }
});

const interceptor = express();
interceptor.post('/compute/:path', interceptPath);
interceptor.get('/nn/:path', interceptPath);
interceptor.get('/nn/:n/:path', interceptPath);
interceptor.get('/nn/:n/:start/:end/:path', interceptPath);

app0.use('/', interceptor);
app0.use('/', proxy({ protocol: 'http', host: 'smqtk', port: 12345 }));
app1.use('/', proxy({ protocol: 'http', host: 'smqtk', port: 12346 }));

createInitialLinks(() => {
  app0.listen(12345, () => {
    app1.listen(12346, () => {
      console.log('Proxy Service Ready');
    });
  });
});

