import sys
import os

for i in range(int(sys.argv[1]), int(sys.argv[2])+1):
    os.system('scancel %d' % i)
