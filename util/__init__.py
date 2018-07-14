import sys
import math
from .utils import ClsFunc

try:
	import c3DSkit
	c3DSkit._confirm()
	libkit = c3DSkit
except:
	import py3DSkit
	libkit = py3DSkit

BOMS = {
	'>': 0xfeff,
	'<': 0xfffe
}

ENDIANS = {
	0xfeff: '>',
	0xfffe: '<'
}

def _Error(errno, type):
	def _Error_Function(cls, msg):
		errormsg = '%s: %s (%d)' % (type, msg, errno)
		if cls.debug:
			raise RuntimeError(errormsg)
		else:
			print(errormsg, file=sys.stderr)
			sys.exit(errno)
	return _Error_Function


def _Warning(errno, type):
	def _Warning_Function(cls, msg):
		print('%s: %s (%d)' % (type, msg, errno), file=sys.stdout)
	return _Warning_Function

class error (ClsFunc):
	debug = False  #-V option, turns errors into exceptions
	UnsupportedFormatError = classmethod(_Error(101, 'UnsupportedFormatError'))
	UnrecognizedFormatError = classmethod(_Error(102, 'UnrecognizedFormatError'))
	UnsupportedCompressionError = classmethod(_Error(103, 'UnsupportedCompressionError'))
	UnknownDataFormatError = classmethod(_Error(104, 'UnknownDataFormatError'))
	UnsupportedDataFormatError = classmethod(_Error(104, 'UnsupportedDataFormatError'))
	UnsupportedVersionError = classmethod(_Error(105, 'UnsupportedVersionError'))
	UnsupportedSettingError = classmethod(_Error(106, 'UnsupportedSettingError'))
	PluginNotFoundError = classmethod(_Error(107, 'PluginNotFoundError'))

	ForgottenArgumentError = classmethod(_Error(201, 'MissingArgumentError'))
	MissingArgumentError = classmethod(_Error(201, 'MissingArgumentError'))
	InvalidInputError = classmethod(_Error(202, 'InvalidInputError'))
	InvalidOptionValueError = classmethod(_Error(203, 'InvalidOptionValueError'))
	UserInterrupt = classmethod(_Error(204, 'UserInterrupt'))
	MissingOptionError = classmethod(_Error(205, 'MissingOptionError'))

	InvalidMagicError = classmethod(_Error(301, 'InvalidMagicError'))
	InvalidSectionError = classmethod(_Error(302, 'InvalidSectionError'))
	HashMismatchError = classmethod(_Error(303, 'HashMismatchError'))
	InvalidFormatError = classmethod(_Error(304, 'InvalidFormatError'))

	NeededDataNotFoundError = classmethod(_Error(401, 'NeededDataNotFoundError'))
	FileNotFoundError = classmethod(_Error(404, 'FileNotFoundError'))
	
	UnrecognizedFormatWarning = classmethod(_Warning(901, 'UnrecognizedFormatWarning'))
	UnsupportedDataFormatWarning = classmethod(_Warning(902, 'UnsupportedDataFormatWarning'))
	InternalCorrectionWarning = classmethod(_Warning(903, 'InternalCorrectionWarning'))
	InvalidInputWarning = classmethod(_Warning(904, 'InvalidInputWarning'))
	InvalidDataWarning = classmethod(_Warning(905, 'InvalidDataWarning'))
	StrangeValueWarning = classmethod(_Warning(906, 'StrangeValueWarning'))
	SettingWarning = classmethod(_Warning(907, 'SettingWarning'))
	def main(self, msg, errno):
		if math.floor(errno / 100) != 9:
			print('Error: %s (%d)' % (msg, errno))
			sys.exit(errno)
		else:
			print('Warning: %s (%d)' % (msg, errno))
			return errno
