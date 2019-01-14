import hashlib
import os
import requests
import sys

# Downloads an imageurl to a location based on its sha1 hash
# Then print out the mapping of the URL to the sha

NUM_RETRIES = 5

if __name__ == '__main__':
    url = sys.argv[1]

    DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '../data')

    i = 0
    while True:
        try:
            r = requests.get(url, stream=True)
        except Exception as ex:
            print("Encountered exception: {}".format(str(ex)))
            sys.exit(3)

        if (r.status_code >= 400 and r.status_code < 500):
            sys.exit(1)
        elif i == NUM_RETRIES:
            sys.exit(2)
        elif not r.ok:
            i += 1
            continue
        else:
            sha1 = hashlib.sha1(r.content).hexdigest()

            try:
                os.makedirs(os.path.join(DATA_DIR, 'CP1_imageset', sha1[0:3]))
            except OSError:
                pass

            with open(os.path.join(DATA_DIR, 'CP1_imageset', sha1[0:3], sha1), 'wb') as outfile:
                outfile.write(r.content)

            print('%s %s' % (url, sha1))

        sys.exit(0)
