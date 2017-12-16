import sys
import math
from .funcops import ClsFunc

BOMS = {
	'>': 0xfeff,
	'<': 0xfffe
}

ENDIANS = {
	0xfeff: '>',
	0xfffe: '<'
}

def _Error(errno):
	def _Error_Function(msg):
		print('Error: %s (%d)' % (msg, errno))
		sys.exit(errno)
	return _Error_Function


def _Warning(errno):
	def _Warning_Function(msg):
		print('Warning: %s (%d)' % (msg, errno))
	return _Warning_Function

class error (ClsFunc):
	UnsupportedFormatError = staticmethod(_Error(101))
	UnrecognizedFormatError = staticmethod(_Error(102))
	UnsupportedCompressionError = staticmethod(_Error(103))
	UnknownDataFormatError = staticmethod(_Error(104))
	UnsupportedVersionError = staticmethod(_Error(105))
	UnsupportedSettingError = staticmethod(_Error(106))
	PluginNotFoundError = staticmethod(_Error(107))
	ForgottenArgumentError = staticmethod(_Error(201))
	InvalidInputError = staticmethod(_Error(202))
	InvalidOptionValueError = staticmethod(_Error(203))
	InvalidMagicError = staticmethod(_Error(301))
	InvalidSectionError = staticmethod(_Error(302))
	HashMismatchError = staticmethod(_Error(303))
	InvalidFormatError = staticmethod(_Error(304))
	NeededDataNotFoundError = staticmethod(_Error(301))
	FileNotFoundError = staticmethod(_Error(404))
	
	UnrecognizedFormatWarning = staticmethod(_Warning(901))
	UnsupportedDataFormatWarning = staticmethod(_Warning(902))
	InternalCorrectionWarning = staticmethod(_Warning(903))
	InvalidInputWarning = staticmethod(_Warning(904))
	def main(self, msg, errno):
		if math.floor(errno / 100) != 9:
			print('Error: %s (%d)' % (msg, errno))
			sys.exit(errno)
		else:
			print('Warning: %s (%d)' % (msg, errno))
			return errno
