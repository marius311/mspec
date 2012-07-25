#This lets your correctly reload this package
try: reload(mspec) 
except: pass
try: reload(utils) 
except: pass

import utils

from mspec import *
