from PyGBASM import *

c = Code(read('pong.z80'))
c.preprocess()
c.assemble()
c.makerom('pong.gb')
