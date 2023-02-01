"""
Analog Plug-in for processing time (esp. real time factor)
"""

__version__   = '$Revision: 6911 $'
__date__      = '$Date: 2008-10-30 15:35:27 +0100 (Thu, 30 Oct 2008) $'


from analog import Collector, Field


class RealTime(Collector):
    id     = 'time'
    name   = 'time'
    fields = [Field('duration', 7, '%7.1f', 's'),
	      Field('CPU',      7, '%7.1f', 's'),
	      Field('rtf',      6, '%6.2f') ]

    def __call__(self, data):
	cpuTime  = sum(data['user time'])
	duration = sum(data['real time'])
	return zip(self.fields, [
	    duration, cpuTime, cpuTime / duration ])

