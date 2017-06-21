const express = require('express');
const request = require('request');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const URL = require('url-parse');
const morgan = require('morgan');

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

const tripPred = (x) => (
  x !== (void 0) &&
  x !== null &&
  isNaN(Number.parseInt(x))
);

const proxy = (options) => {
  const result = (req, res, next) => {
    let path = req.originalUrl;
    if (path[0] === '/') {
      path = path.slice(1);
    }

    let { baseURL, rewrittenURI } = req.params;
    if (baseURL && rewrittenURI) {
      console.log('  REWRITING PATH');
      console.log('    BEFORE: ' + path);
      path = [baseURL, rewrittenURI].join('/');
      console.log('    AFTER : ' + path);
    }

    let { protocol, host, port } = options;

    let newLocation = protocol + '://' + host + ':' + port + '/' + path;
    console.log('  PROXYING TO ' + newLocation);
    let proxyReq = request(newLocation);
    req.pipe(proxyReq).pipe(res);
  };

  return result;
};

const interceptPath = (req, res, next) => {
  console.log('  INTERCEPTING');
  let { n, start, end, path } = req.params;
  let uri = [];
  let tripped = tripPred(n);
  if (tripped) { uri.push(n); } else { tripped = tripPred(start); }
  if (tripped) { uri.push(start); } else { tripped = tripPred(end); }
  if (tripped) { uri.push(end); }
  uri.push(path);

  let rest = req.params[0];
  if (rest[0] === '/') {
    rest = rest.slice(1);
  }
  uri = uri.concat(rest.split('/')).join('/');

  let baseUrl = req.originalUrl.slice(0, req.originalUrl.length - uri.length);

  if (baseUrl[baseUrl.length - 1] === '/') {
    baseUrl = baseUrl.slice(0, baseUrl.length - 1);
  }

  if (baseUrl[0] === '/') {
    baseUrl = baseUrl.slice(1);
  }

  if (!uri.match(/^[a-z]+:\/\//)) {
    uri = 'http://' + uri;
  }

  const done = (() => {
    let counter = 0;
    return () => {
      if (counter < 2) {
        ++counter;
        if (counter === 2) {
          next();
        }
      }
    };
  })();

  createNewLink(request(uri), done);

  let url = new URL(uri);
  if (url.hostname === 'localhost') {
    console.log('  REWRITING PATH HOSTNAME');
    console.log('    BEFORE: ' + uri);
    url.set('hostname', 'wrapper');
    uri = url.toString();
    console.log('    AFTER : ' + uri);
  }

  if (uri[uri.length - 1] === '/') {
    uri = uri.slice(0, uri.length - 1);
  }

  req.params.rewrittenURI = uri;
  req.params.baseURL = baseUrl;

  console.log('    BASE URL: ' + baseUrl);
  done();
};

const app0 = express();
const app1 = express();

app0.use(morgan('dev', { immediate: true }));
app1.use(morgan('dev', { immediate: true }));

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

let middleWare = [
  interceptPath,
  proxy({ protocol: 'http', host: 'smqtk', port: 12345 })
];

app0.post('/compute/:path', middleWare);
app0.get('/nn/:n/:start/:end/:path*', middleWare);
app0.get('/nn/:n/:start/:path*', middleWare);
app0.get('/nn/:n/:path*', middleWare);
app0.get('/nn/:path*', middleWare);

app0.route('*').all(middleWare[1]);
app1.route('*').all(proxy({ protocol: 'http', host: 'smqtk', port: 12346 }))
createInitialLinks(() => {
  app0.listen(12345, () => {
    app1.listen(12346, () => {
      console.log('Proxy Service Ready');
    });
  });
});
