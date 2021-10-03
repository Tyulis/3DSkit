# -*- coding:utf-8 -*-
# rawutil.py
# A single-file, pure-python module to manage binary data

import io
import sys
import math
import copy
import builtins
import binascii
import collections

__version__ = "2.7.1"

ENDIANNAMES = {
	"=": sys.byteorder,
	"@": sys.byteorder,
	">": "big",
	"!": "big",
	"<": "little",
}

ENDIANMARKS = {
	"little": "<",
	"big": ">",
}

def bin(val, align=0):
	if isinstance(val, int):
		return builtins.bin(val).lstrip('0b').zfill(align)
	elif type(val) in (bytes, bytearray, list, tuple):
		return ''.join([builtins.bin(b).lstrip('0b').zfill(8) for b in val]).zfill(align)
	else:
		raise TypeError('Int, bytes or bytearray object is needed')

def hex(val, align=0):
	if isinstance(val, int):
		return builtins.hex(val).lstrip('0x').zfill(align)
	else:
		return binascii.hexlify(bytes(val)).decode('ascii').zfill(align)


def hextoint(hx):
	return int(hx, 16)


def hextobytes(hx):
	if type(hx) == str:
		hx = hx.encode('ascii')
	return binascii.unhexlify(hx)


class FormatError (Exception):
	def __init__(self, message, format=None, subformat=None, position=None):
		self.message = message
		self.format = format
		self.subformat = subformat
		self.position = position

	def __str__(self):
		message = ""
		if self.format is not None:
			message += "In format '" + self.format + "'"
		if self.subformat is not None:
			message += ", in subformat '" + self.subformat + "'"
		if self.position is not None:
			message += ", at position " + str(self.position)
		message += " : " + self.message
		return message

class OperationError (Exception):
	def __init__(self, message, format=None, subformat=None):
		self.message = message
		self.format = format
		self.subformat = subformat

	def __str__(self):
		message = ""
		if self.format is not None:
			message += "In format '" + self.format + "'"
		if self.subformat is not None:
			message += ", in subformat '" + self.subformat + "'"
		message += " : " + self.message
		return message

class _Reference (object):
	REFERENCE_TYPES = {0: None, 1: "relative", 2: "absolute", 3: "external"}
	def __init__(self, type, value):
		self.type = type
		self.value = value

	def copy(self):
		return _Reference(self.type, self.value)

	def __str__(self):
		return "Ref(" + self.REFERENCE_TYPES[self.type] + ", " + str(self.value) + ")"

	def __repr(self):
		return "_Reference(" + str(self.type) + ", " + str(self.value) + ")"

class _Token (object):
	def __init__(self, count, type, content):
		self.count = count
		self.type = type
		self.content = content

	def copy(self):
		if isinstance(self.count, _Reference):
			count = self.count.copy()
		else:
			count = self.count
		if self.content is None:
			content = None
		else:
			content = [token.copy() for token in self.content]
		return _Token(count, self.type, content)

	def __str__(self):
		return "(" + str(self.count) + ", " + self.type + ")"

	def __repr__(self):
		return "_Token(" + repr(self.count) + ", '" + self.type + "', " + repr(self.content) + ")"


_GROUP_CHARACTERS = {"(": ")", "[": "]", "{": "}"}
_NO_MULTIPLE = "{|$"
_END_STRUCTURE = "$"
_INTEGER_ELEMENTS = {  # (signed, size in bytes)
	"b": (True, 1), "B": (False, 1), "h": (True, 2), "H": (False, 2),
	"u": (True, 3), "U": (False, 3), "i": (True, 4), "I": (False, 4),
	"l": (True, 4), "L": (False, 4), "q": (True, 8), "Q": (False, 8),
}
_FLOAT_ELEMENTS = {  # (size in bytes, exponent bits, factor bits, -exponent, maxvalue)
	"e": (2, 5, 10, 15), "f": (4, 8, 23, 127), "d": (8, 11, 52, 1023), "F": (16, 15, 112, 16383),
}
_STRUCTURE_CHARACTERS = {
	"?": 1, "b": 1, "B": 1, "h": 2, "H": 2, "u": 3, "U": 3,
	"i": 4, "I": 4, "l": 4, "L": 4, "q": 8, "Q": 8, "e": 2, "f": 4, "d": 8,
	"F": 16, "c": 1, "s": 1, "n": None, "X": 1, "|": 0, "a": -1, "x": 1, "$": None,
}

class Struct (object):
	def __init__(self, format="", names=None):
		self.names = None
		if isinstance(format, Struct):
			self.format = format.format
			self.byteorder = format.byteorder
			self.forcebyteorder = format.forcebyteorder
			self.tokens = copy.deepcopy(format.tokens)
			self.names = format.names
		elif len(format) > 0:
			self.format = format
			self.byteorder = sys.byteorder
			self.forcebyteorder = False
			self.parse_struct(format)
		else:
			self.format = format
			self.byteorder = sys.byteorder
			self.forcebyteorder = False
			self.tokens = []

		if hasattr(names, '_fields') and hasattr(names, '_asdict'):  #trying to recognize a namedtuple
			self.names = names
		elif names is not None:
			self.names = collections.namedtuple('RawutilNameSpace', names)

	def parse_struct(self, format):
		format = self.preprocess(format)
		if format[0] in tuple(ENDIANNAMES.keys()):
			self.byteorder = ENDIANNAMES[format[0]]
			self.forcebyteorder = True
			self.format = format = format[1:]

		self.tokens = self.parse_substructure(format)

	def preprocess(self, format):
		format = format.strip().replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
		while "'" in format:
			start = format.find("'")
			end = format.find("'", start + 1)
			format = format[:start] + format[end + 1:]
		return format

	def parse_substructure(self, format):
		tokens = []
		ptr = 0
		while ptr < len(format):
			startptr = ptr
			# References
			if format[ptr] == "/":
				ptr += 1
				if format[ptr] == "p":  # Relative
					reftype = 1
					ptr += 1
				else:  # Absolute
					reftype = 2
			elif format[ptr] == "#":  # Extern reference
				reftype = 3
				ptr += 1
			else:  # Normal count
				reftype = None

			# Parsing count
			countstr = ""
			while format[ptr].isdigit():
				countstr += format[ptr]
				ptr += 1

			if len(countstr) == 0:
				if reftype is not None:
					raise FormatError("No reference number", self.format, format, startptr)
				count = 1
			else:
				count = int(countstr)
				if reftype is not None:
					count = _Reference(reftype, count)

			# Groups
			if format[ptr] in _GROUP_CHARACTERS:
				openchar = format[ptr]
				closechar = _GROUP_CHARACTERS[format[ptr]]
				subformat = ""
				ptr += 1
				level = 1
				while level > 0:
					subformat += format[ptr]
					if format[ptr] == openchar:
						level += 1
					elif format[ptr] == closechar:
						level -= 1
					ptr += 1
				subformat = subformat[:-1]
				type = openchar
				content = self.parse_substructure(subformat)
			else:  # Standard structure elements
				if format[ptr] not in _STRUCTURE_CHARACTERS:
					raise FormatError("Unrecognised character '" + format[ptr] + "'", self.format, format, startptr)
				else:
					type = format[ptr]
				content = None
				ptr += 1
			if count != 1 and type in _NO_MULTIPLE:
				raise FormatError("'" + type + "' elements should not be multiple", self.format, format, startptr)
			if ptr < len(format) and type in _END_STRUCTURE:
				raise FormatError("'" + type + "' terminate the structure, there should be nothing else after them", self.format, format, startptr)
			token = _Token(count, type, content)
			tokens.append(token)
		return tokens

	def pprint(self, tokens=None):
		out = ""
		if tokens is None:
			tokens = self.tokens
		for token in tokens:
			out += str(token) + "\n"
			if token.content is not None:
				substruct = self.pprint(token.content)
				out += "\n".join(["\t" + line for line in substruct.splitlines()]) + "\n"
		return out

	def setbyteorder(self, byteorder):
		if byteorder in ENDIANNAMES:
			byteorder = ENDIANNAMES[byteorder]
		self.forcebyteorder = True
		self.byteorder = byteorder

	def asbyteorder(self, byteorder):
		copy = Struct(self)
		copy.setbyteorder(byteorder)
		return copy

	def unpack(self, data, names=None, refdata=()):
		if hasattr(data, "read") and hasattr(data, "tell"):  # From file-like object
			unpacked = self._unpack_file(data, self.tokens, refdata)
		else:  # From bytes-like objet
			unpacked = self._unpack_file(io.BytesIO(data), self.tokens, refdata)

		if hasattr(names, '_fields') and hasattr(names, '_asdict'):  #trying to recognize a namedtuple
			unpacked = names(unpacked)
		elif names is not None:
			unpacked = collections.namedtuple('RawutilNameSpace', names)(*unpacked)
		elif self.names is not None:
			unpacked = self.names(*unpacked)
		return unpacked

	def unpack_from(self, data, offset=None, names=None, refdata=(), getptr=False):
		if hasattr(data, "read") and hasattr(data, "tell"):  # From file-like object
			if offset is not None:
				data.seek(offset)
			unpacked = self._unpack_file(data, self.tokens, refdata)
		else:  # From bytes-like objet
			data = io.BytesIO(data)
			if offset is not None:
				data.seek(offset)
			unpacked = self._unpack_file(data, self.tokens, refdata)

		if hasattr(names, '_fields') and hasattr(names, '_asdict'):  #trying to recognize a namedtuple
			unpacked = names(unpacked)
		elif names is not None:
			unpacked = collections.namedtuple('RawutilNameSpace', names)(*unpacked)
		elif self.names is not None:
			unpacked = self.names(*unpacked)

		if getptr:
			return unpacked, data.tell()
		else:
			return unpacked

	def pack(self, *data, refdata=()):
		data = list(data)

		if hasattr(data[-1], "write") and hasattr(data, "seek"):  # Into file-like object
			out = data.pop(-1)
			self._pack_file(out, data, refdata)
		else:
			out = io.BytesIO()
			self._pack_file(out, data, refdata)
			out.seek(0)
			return out.read()

	def pack_into(self, buffer, offset, *data, refdata=()):
		out = io.BytesIO()
		self._pack_file(out, data, refdata)
		out.seek(0)
		packed = out.read()
		buffer[offset: offset + len(packed)] = packed

	def pack_file(self, file, *data, position=None, refdata=()):
		if position is not None:
			file.seek(position)
		self._pack_file(file, data, refdata)

	def iter_unpack(self, data, names=None, refdata=()):
		if hasattr(data, "read") and hasattr(data, "tell"):  # From file-like object
			buffer = data
		else:
			buffer = io.BytesIO(data)
		pos = buffer.tell()
		buffer.seek(0, 2)
		end = buffer.tell()
		buffer.seek(pos)
		while buffer.tell() < end:
			unpacked = self._unpack_file(buffer, self.tokens, refdata)
			if hasattr(names, '_fields') and hasattr(names, '_asdict'):  #trying to recognize a namedtuple
				unpacked = names(unpacked)
			elif names is not None:
				unpacked = collections.namedtuple('RawutilNameSpace', names)(*unpacked)
			elif self.names is not None:
				unpacked = self.names(*unpacked)
			yield unpacked

	def calcsize(self, refdata=None, tokens=None):
		if tokens is None:
			tokens = self.tokens
		size = 0
		alignref = 0
		for token in tokens:
			if isinstance(token.count, _Reference):
				if token.count.type == 3 and refdata is not None:
					count = refdata[token.count.value]
				else:
					raise FormatError("Impossible to compute the size of a structure with references", self.format)
			else:
				count = token.count

			if token.type in "[(":
				size += count * self.calcsize(refdata, token.content)
			elif token.type == "{":
				raise FormatError("Impossible to compute the size of a structure with {} iterators", self.format)
			elif token.type == "|":
				alignref = size
			else:
				elementsize = _STRUCTURE_CHARACTERS[token.type]
				if elementsize is None:
					raise FormatError("Impossible to compute the size of a structure with '" + token.type + "' elements", self.format)
				elif elementsize == -1:
					refdistance = size - alignref
					padding = count - (refdistance % count or count)
					size += padding
				else:
					size += count * elementsize
		return size


	def _read(self, data, length=-1):
		read = data.read(length)
		if len(read) < length:
			raise IndexError("Not enough data to read")
		else:
			return read

	def _resolve_count(self, count, unpacked, refdata):
		if isinstance(count, _Reference):
			try:
				if count.type == 1:  # Relative
					value = unpacked[-count.value]
				elif count.type == 2:  # Absolute
					value = unpacked[count.value]
				elif count.type == 3:  # External
					value = refdata[count.value]

				if not isinstance(value, int):
					raise OperationError("Count from " + _Reference.REFERENCE_TYPES[count.type] + " reference index " + str(count.value) + " is not an integer", self.format)
				return value
			except IndexError:
				raise OperationError("Bad " + _Reference.REFERENCE_TYPES[count.type] + "reference value : " + str(count.value), self.format)
		else:
			return count

	def _convert_mantissa(self, mantissa, size):
		result = 0
		factor = 2**(-size)
		for _ in range(size):
			result += (mantissa & 1) * factor
			mantissa >>= 1
			factor *= 2
		return result

	def _build_float(self, value, exponentsize, mantissasize, bias):
		sign = (0 if value >= 0 else 1)
		value = abs(value)

		if value == 0:
			exponent = 0
			mantissa = 0
		elif value == math.inf:
			exponent = (1 << exponentsize) - 1
			mantissa = 0
		elif value == math.nan:
			exponent = (1 << exponentsize) - 1
			mantissa = (1 << mantissasize) - 1
		else:
			exponent = 0
			exponentrange = (1 << (exponentsize - 1)) - 1

			normalised = value
			while normalised >= 2 and exponent <= exponentrange:
				normalised /= 2
				exponent += 1
			while normalised < 1 and -exponentrange + 1 <= exponent:
				normalised *= 2
				exponent -= 1

			if 2 ** (-mantissasize) <= normalised < 1:
				exponent = -exponentrange
				normalised /= 2
			elif 1 <= normalised < 2:
				normalised -= 1
			else:
				raise ValueError("Floating-point value " + str(value) + " is out of range for " + str(exponentsize + mantissasize + 1) + " bits float")

			mantissa = 0
			for _ in range(mantissasize):
				normalised *= 2
				intpart = int(normalised)
				mantissa = (mantissa << 1) | intpart
				normalised -= intpart
			if 0.5 < normalised < 1 or (normalised == 0.5 and mantissa & 1):  # rounding to the nearest, ties to even
				mantissa += 1

			exponent += bias

		return sign, exponent, mantissa


	def _unpack_file(self, data, tokens, refdata):
		alignref = data.tell()
		unpacked = []

		for token in tokens:
			count = self._resolve_count(token.count, unpacked, refdata)

			try:
				# Groups
				if token.type == "(":
					multigroup = []
					for i in range(count):
						subgroup = self._unpack_file(data, token.content, refdata)
						multigroup.extend(subgroup)
					unpacked.append(multigroup)
				elif token.type == "[":
					sublist = []
					for i in range(count):
						subgroup = self._unpack_file(data, token.content, refdata)
						sublist.append(subgroup)
					unpacked.append(sublist)
				elif token.type == "{":
					sublist = []
					while True:
						try:
							subgroup = self._unpack_file(data, token.content, refdata)
						except OperationError:
							break
						sublist.append(subgroup)
					unpacked.append(sublist)
				# Control
				elif token.type == "|":
					alignref = data.tell()
				elif token.type == "a":
					refdistance = data.tell() - alignref
					padding = count - (refdistance % count or count)
					data.seek(padding, 1)
				elif token.type == "$":
					unpacked.append(self._read(data))
				# Elements
				elif token.type in _INTEGER_ELEMENTS:
					signed, size = _INTEGER_ELEMENTS[token.type]
					groupdata = self._read(data, size * count)
					for i in range(count):
						unpacked.append(int.from_bytes(groupdata[i*size: (i+1)*size], byteorder=self.byteorder, signed=signed))
				elif token.type in _FLOAT_ELEMENTS:
					size, exponentsize, mantissasize, bias = _FLOAT_ELEMENTS[token.type]
					groupdata = self._read(data, size * count)
					for i in range(count):
						elementdata = groupdata[i*size: (i+1)*size]

						encoded = int.from_bytes(elementdata, byteorder=self.byteorder, signed=False)
						sign = (-1 if (encoded >> (exponentsize + mantissasize)) else 1)
						baseexponent = ((encoded >> mantissasize) & ((1 << exponentsize) - 1))
						exponent = baseexponent - bias
						mantissa = (encoded & ((1 << mantissasize) - 1))
						if baseexponent == 0 and mantissa == 0:
							decoded = sign * 0.0
						elif baseexponent == 0 and mantissa != 0:
							factor = self._convert_mantissa(mantissa, mantissasize)
							decoded = sign * factor * (2 ** (exponent + 1))
						elif baseexponent == (1 << exponentsize) - 1 and mantissa == 0:
							decoded = sign * math.inf
						elif baseexponent == (1 << exponentsize) - 1 and mantissa != 0:
							decoded = math.nan
						else:
							factor = self._convert_mantissa(mantissa, mantissasize) + 1
							decoded = sign * factor * (2 ** exponent)
						unpacked.append(decoded)
				elif token.type == "x":
					data.seek(count, 1)
				elif token.type == "?":
					elementdata = self._read(data, count)
					unpacked.extend([bool(byte) for byte in elementdata])
				elif token.type == "c":
					elementdata = self._read(data, count)
					unpacked.extend([bytes((byte, )) for byte in elementdata])
				elif token.type == "s":
					unpacked.append(self._read(data, count))
				elif token.type == "n":
					for _ in range(count):
						string = b""
						while True:
							char = self._read(data, 1)
							if char == b"\x00":
								break
							else:
								string += char
						unpacked.append(string)
				elif token.type == "X":
					elementdata = self._read(data, count)
					unpacked.append(elementdata.hex())
			except IndexError:
				raise OperationError("No data remaining to read element '" + token.type + "'", self.format)

		return unpacked

	def _pack_file(self, out, data, refdata, tokens=None):
		if tokens is None:
			tokens = self.tokens
		position = 0
		alignref = out.tell()
		for token in tokens:
			count = self._resolve_count(token.count, data[:position], refdata)
			try:
				# Groups
				if token.type == "(":
					grouppos = 0
					for _ in range(count):
						grouppos += self._pack_file(out, data[position][grouppos:], refdata, token.content)
					position += 1
				elif token.type == "[":
					for _, group in zip(range(count), data[position]):
						self._pack_file(out, group, refdata, token.content)
					position += 1
				elif token.type == "{":
					for group in data[position]:
						self._pack_file(out, group, refdata, token.content)
					position += 1
				# Control
				elif token.type == "|":
					alignref = out.tell()
				elif token.type == "a":
					refdistance = out.tell() - alignref
					padding = count - (refdistance % count or count)
					out.write(b"\x00" * padding)
				elif token.type == "$":
					out.write(data[position])
					position += 1
				elif token.type in _INTEGER_ELEMENTS:
					signed, size = _INTEGER_ELEMENTS[token.type]
					elementdata = b""
					for _ in range(count):
						elementdata += data[position].to_bytes(size, byteorder=self.byteorder, signed=signed)
						position += 1
					out.write(elementdata)
				elif token.type in _FLOAT_ELEMENTS:
					size, exponentsize, mantissasize, bias = _FLOAT_ELEMENTS[token.type]
					elementdata = b""
					for _ in range(count):
						decoded = data[position]
						position += 1

						sign, exponent, mantissa = self._build_float(decoded, exponentsize, mantissasize, bias)
						encoded = (sign << (exponentsize + mantissasize)) | (exponent << mantissasize) | mantissa
						elementdata += encoded.to_bytes(size, byteorder=self.byteorder, signed=False)
					out.write(elementdata)
				elif token.type == "x":
					out.write(b"\x00" * count)
				elif token.type == "?":
					elementdata = bytes(data[position: position + count])
					out.write(elementdata)
					position += count
				elif token.type == "c":
					elementdata = b"".join(data[position: position + count])
					out.write(elementdata)
					position += count
				elif token.type == "s":
					string = self._encode_string(data[position])
					if len(string) != count:
						raise OperationError("Length of structure element 's' (" + str(count) + " and data '" + repr(data[position]) + "' do not match", self.format)
					out.write(string)
					position += 1
				elif token.type == "n":
					string = self._encode_string(data[position])
					out.write(string + b"\x00")
					position += 1
				elif token.type == "X":
					out.write(bytes.fromhex(data[position]))
					position += 1
			except IndexError:
				raise OperationError("No data remaining to pack into element '" + token.type + "'", self.format)
		return position

	def _encode_string(self, data):
		try:
			string = data.encode("utf-8")
		except (AttributeError, UnicodeDecodeError):
			string = data
		return string

	def _count_to_format(self, count):
		if count == 1:
			return ""
		elif isinstance(count, _Reference):
			if count.type == 1:  # relative
				return "/p" + str(count.value)
			elif count.type == 2:  # absolute
				return "/" + str(count.value)
			elif count.type == 3:  # external
				return "#" + str(count.value)
		else:
			return str(count)

	def _tokens_to_format(self, tokens):
		format = ""
		for token in tokens:
			if token.type in _GROUP_CHARACTERS:
				subformat = self._tokens_to_format(token.content)
				format += self._count_to_format(token.count) + token.type + subformat + _GROUP_CHARACTERS[token.type] + " "
			else:
				format += self._count_to_format(token.count) + token.type + " "
		return format.strip()

	def _max_external_reference(self, tokens):
		maxref = -1
		for token in tokens:
			if isinstance(token.count, _Reference):
				if token.count.type == 3:
					if token.count.value > maxref:
						maxref = token.count.value
			if token.content is not None:
				submax = self._max_external_reference(token.content)
				if submax > maxref:
					maxref = submax
		return maxref

	def _fix_external_references(self, tokens, leftexternals):
		for token in tokens:
			if isinstance(token.count, _Reference):
				if token.count.type == 3:
					token.count.value += leftexternals
			if token.content is not None:
				self._fix_external_references(token.content, leftexternals)

	def _add_structs(self, lefttokens, righttokens):
		leftsize = len(lefttokens)
		leftexternals = self._max_external_reference(lefttokens) + 1

		outtokens = []
		for token in lefttokens:
			if token.type in ("{", "$"):
				raise FormatError("'" + token.type + ("}" if token.type == "{" else "") + "' forces the end of the structure, you can’t add or multiply structures if it causes those elements to be in the middle of the resulting structure")
			outtokens.append(token.copy())

		for token in righttokens:
			newtoken = token.copy()
			if isinstance(newtoken.count, _Reference):
				if newtoken.count.type == 2:  # absolute
					newtoken.count.value += leftsize
				elif newtoken.count.type == 3:  # external
					newtoken.count.value += leftexternals
			if newtoken.content is not None:
				self._fix_external_references(newtoken.content, leftexternals)
			outtokens.append(newtoken)
		outformat = self._tokens_to_format(outtokens)
		return outtokens, outformat

	def _multiply_struct(self, tokens, num):
		blocksize = len(tokens)
		blockexternals = self._max_external_reference(tokens) + 1

		size = 0
		externals = 0
		outtokens = []
		for _ in range(num):
			for token in tokens:
				if token.type in ("{", "$"):
					raise FormatError("'" + token.type + ("}" if token.type == "{" else "") + "' forces the end of the structure, you can’t add or multiply structures if it causes those elements to be in the middle of the resulting structure")
				newtoken = token.copy()
				if isinstance(newtoken.count, _Reference):
					if newtoken.count.type == 2:  # absolute
						newtoken.count.value += size
					elif newtoken.count.type == 3:  # external
						newtoken.count.value += externals
				if newtoken.content is not None:
					self._fix_external_references(newtoken.content, externals)
				outtokens.append(newtoken)
			size += blocksize
			externals += blockexternals
		outformat = self._tokens_to_format(outtokens)
		return outtokens, outformat


	def __add__(self, stct):
		if not isinstance(stct, Struct):
			stct = Struct(stct)
		newtokens, newformat = self._add_structs(self.tokens, stct.tokens)

		newstruct = Struct()
		newstruct.format = newformat
		newstruct.tokens = newtokens
		if self.forcebyteorder:
			newstruct.setbyteorder(self.byteorder)
		return newstruct

	def __iadd__(self, stct):
		if not isinstance(stct, Struct):
			stct = Struct(stct)
		newtokens, newformat = self._add_structs(self.tokens, stct.tokens)

		self.tokens = newtokens
		self.format = newformat
		return newstruct

	def __radd__(self, stct):
		if not isinstance(stct, Struct):
			stct = Struct(stct)
		newtokens, newformat = self._add_structs(stct.tokens, self.tokens)

		newstruct = Struct()
		if stct.forcebyteorder:
			newstruct.setbyteorder(stct.byteorder)
		newstruct.tokens = newtokens
		newstruct.format = newformat
		return newstruct

	def __mul__(self, n):
		newtokens, newformat = self._multiply_struct(self.tokens, n)

		newstruct = Struct()
		if self.forcebyteorder:
			newstruct.setbyteorder(self.byteorder)
		newstruct.tokens = newtokens
		newstruct.format = newformat
		return newstruct

	def __imul__(self, n):
		newtokens, newformat = self._multiply_struct(self.tokens, n)
		self.tokens = newtokens
		self.format = newformat
		return self

	def __rmul__(self, n):
		newtokens, newformat = self._multiply_struct(self.tokens, n)

		newstruct = Struct()
		if stct.forcebyteorder:
			newstruct.setbyteorder(stct.byteorder)
		newstruct.tokens = newtokens
		newstruct.format = newformat
		return newstruct

	def __repr__(self):
		return "Struct(\"" + self.format + "\")"

	def __str__(self):
		return self.__repr__()


class TypeUser (object):
	def __init__(self, byteorder="@"):
		self.byteorder = ENDIANNAMES[byteorder]

	def unpack(self, structure, data, names=None, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.unpack(data, names, refdata)

	def unpack(self, structure, data, names=None, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.unpack(data, names, refdata)

	def unpack_from(self, structure, data, offset=None, names=None, refdata=(), getptr=False):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.unpack_from(data, offset, names, refdata, getptr)

	def iter_unpack(self, structure, data, names=None, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.iter_unpack(data, names, refdata)

	def pack(self, structure, *data, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.pack(*data, refdata=refdata)

	def pack_into(self, structure, buffer, offset, *data, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.pack_into(buffer, offset, *data, refdata=refdata)

	def pack_file(self, structure, file, *data, position=None, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.pack_file(file, *data, position=position, refdata=refdata)

	def calcsize(self, structure, refdata=()):
		stct = Struct(structure)
		if not stct.forcebyteorder:
			stct.setbyteorder(self.byteorder)
		return stct.calcsize()

def _readermethod(stct):
		def _TypeReader_method(self, data, ptr=None):
			(result, ), ptr = self.unpack_from(stct, data, ptr, getptr=True)
			return result, ptr
		return _TypeReader_method

class TypeReader (TypeUser):
	bool = _readermethod(Struct("?"))
	int8 = _readermethod(Struct("b"))
	uint8 = _readermethod(Struct("B"))
	int16 = _readermethod(Struct("h"))
	uint16 = _readermethod(Struct("H"))
	int24 = _readermethod(Struct("u"))
	uint24 = _readermethod(Struct("U"))
	int32 = _readermethod(Struct("i"))
	uint32 = _readermethod(Struct("I"))
	int64 = _readermethod(Struct("q"))
	uint64 = _readermethod(Struct("Q"))
	half = float16 = _readermethod(Struct("e"))
	single = float = float32 = _readermethod(Struct("f"))
	double = float64 = _readermethod(Struct("d"))
	quad = float128 = _readermethod(Struct("F"))
	string = _readermethod(Struct("n"))

	def tobits(self, n, align=8):
		return [int(bit) for bit in bin(n, align)]

	def bit(self, n, bit, length=1):
		mask = ((2 ** length) - 1) << bit
		return (n & mask) >> (bit - length)

	def nibbles(self, n):
		return (n >> 4, n & 0xf)

	def signed_nibbles(self, n):
		high = (n >> 4)
		if high >= 8:
			high -= 16
		low = (n & 0xf)
		if low >= 8:
			low -= 16
		return high, low

	def utf16string(self, data, ptr):
		subdata = data[ptr:]
		s = []
		zeroes = 0
		for i, c in enumerate(subdata):
			if c == 0:
				zeroes += 1
			else:
				zeroes = 0
			s.append(c)
			if zeroes >= 2 and i % 2 == 1:
				break
		endian = 'le' if self.byteorder == 'little' else 'be'
		return bytes(s[:-2]).decode('utf-16-%s' % endian), ptr + i

def _writermethod(stct):
		def _TypeWriter_method(self, data, out=None):
			if out is None:
				return self.pack(stct, data)
			else:
				self.pack(stct, data, out)
		return _TypeWriter_method

class TypeWriter (TypeUser):
	bool = _writermethod(Struct("?"))
	int8 = _writermethod(Struct("b"))
	uint8 = _writermethod(Struct("B"))
	int16 = _writermethod(Struct("h"))
	uint16 = _writermethod(Struct("H"))
	int24 = _writermethod(Struct("u"))
	uint24 = _writermethod(Struct("U"))
	int32 = _writermethod(Struct("i"))
	uint32 = _writermethod(Struct("I"))
	int64 = _writermethod(Struct("q"))
	uint64 = _writermethod(Struct("Q"))
	half = float16 = _writermethod(Struct("e"))
	single = float = float32 = _writermethod(Struct("f"))
	double = float64 = _writermethod(Struct("d"))
	quad = float128 = _writermethod(Struct("F"))

	def nibbles(self, high, low):
		return (high << 4) + (low & 0xf)

	def signed_nibbles(self, high, low):
		if high < 0:
			high += 16
		if low < 0:
			low += 16
		return (high << 4) + (low & 0xf)

	def string(self, data, align=0, out=None):
		if isinstance(data, str):
			s = data.encode('utf-8')
		if align < len(s) + 1:
			align = len(s) + 1
		res = struct.pack('%s%ds' % (self.byteorder, align), s)
		if out is None:
			return res
		else:
			out.write(res)

	def utf16string(self, data, align=0, out=None):
		endian = 'le' if self.byteorder == 'little' else 'be'
		s = data.encode('utf-16-%s' % endian) + b'\x00\x00'
		if align < len(s) + 2:
			align = len(s) + 2
		res = struct.pack('%s%ds' % (self.byteorder, align), s)
		if out is None:
			return res
		else:
			out.write(res)

	def pad(self, num):
		return b'\x00' * num

	def align(self, data, alignment):
		if isinstance(data, int):
			length = data
		else:
			length = len(data)
		padding = alignment - (length % alignment or alignment)
		return b'\x00' * padding


class StructurePack (object):
	def __init__(self, **structs):
		self.structs = {}
		for name, struct in structs.items():
			if not isinstance(struct, Struct):
				struct = Struct(struct)
			self.structs[name] = struct

	def asbyteorder(self, byteorder, force=False):
		if byteorder in ENDIANNAMES:
			byteorder = ENDIANNAMES[byteorder]
		newpack = self.copy()
		for name, struct in newpack.structs.items():
			if (struct.forcebyteorder and force) or not struct.forcebyteorder:
				struct.setbyteorder(byteorder)
		return newpack

	def copy(self):
		newpack = StructurePack()
		for name, struct in self.structs.items():
			newpack.structs[name] = Struct(struct)
		return newpack

	def __getattr__(self, attr):
		return self.structs[attr]

def unpack(structure, data, names=None, refdata=()):
	stct = Struct(structure)
	return stct.unpack(data, names, refdata)

def unpack_from(structure, data, offset=None, names=None, refdata=(), getptr=False):
	stct = Struct(structure)
	return stct.unpack_from(data, offset, names, refdata, getptr)

def iter_unpack(structure, data, names=None, refdata=()):
	stct = Struct(structure)
	return stct.iter_unpack(data, names, refdata)

def pack(structure, *data, refdata=()):
	stct = Struct(structure)
	return stct.pack(*data, refdata=refdata)

def pack_into(structure, buffer, offset, *data, refdata=()):
	stct = Struct(structure)
	return stct.pack_into(buffer, offset, *data, refdata=refdata)

def pack_file(structure, file, *data, position=None, refdata=()):
	stct = Struct(structure)
	return stct.pack_file(file, *data, position=position, refdata=refdata)

def calcsize(structure, refdata=()):
	stct = Struct(structure)
	return stct.calcsize(refdata=refdata)
