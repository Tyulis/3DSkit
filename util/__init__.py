import sys
import math
from .utils import ClsFunc

BOMS = {
	'>': 0xfeff,
	'<': 0xfffe
}

ENDIANS = {
	0xfeff: '>',
	0xfffe: '<'
}

def _Error(errno):
	def _Error_Function(cls, msg):
		errormsg = 'Error: %s (%d)' % (msg, errno)
		if cls.debug:
			raise RuntimeError(errormsg)
		else:
			print(errormsg, file=sys.stderr)
			sys.exit(errno)
	return _Error_Function


def _Warning(errno):
	def _Warning_Function(cls, msg):
		print('Warning: %s (%d)' % (msg, errno), file=sys.stdout)
	return _Warning_Function

class error (ClsFunc):
	debug = False  #-V option, turns errors into exceptions
	UnsupportedFormatError = classmethod(_Error(101))
	UnrecognizedFormatError = classmethod(_Error(102))
	UnsupportedCompressionError = classmethod(_Error(103))
	UnknownDataFormatError = classmethod(_Error(104))
	UnsupportedVersionError = classmethod(_Error(105))
	UnsupportedSettingError = classmethod(_Error(106))
	PluginNotFoundError = classmethod(_Error(107))
	ForgottenArgumentError = classmethod(_Error(201))
	InvalidInputError = classmethod(_Error(202))
	InvalidOptionValueError = classmethod(_Error(203))
	InvalidMagicError = classmethod(_Error(301))
	InvalidSectionError = classmethod(_Error(302))
	HashMismatchError = classmethod(_Error(303))
	InvalidFormatError = classmethod(_Error(304))
	NeededDataNotFoundError = classmethod(_Error(301))
	FileNotFoundError = classmethod(_Error(404))
	
	UnrecognizedFormatWarning = classmethod(_Warning(901))
	UnsupportedDataFormatWarning = classmethod(_Warning(902))
	InternalCorrectionWarning = classmethod(_Warning(903))
	InvalidInputWarning = classmethod(_Warning(904))
	InvalidDataWarning = classmethod(_Warning(905))
	def main(self, msg, errno):
		if math.floor(errno / 100) != 9:
			print('Error: %s (%d)' % (msg, errno))
			sys.exit(errno)
		else:
			print('Warning: %s (%d)' % (msg, errno))
			return errno
