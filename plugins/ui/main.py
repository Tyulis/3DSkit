# -*- coding: utf-8 -*-
import os
from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import *
from util.fileops import *
from pack import pack, formats

ENDIANS = {'little': '<', 'big': '>'}

class _3DSkitUI (Tk):
	def __init__(self, options):
		super().__init__()
		self.main(options)
	
	def parse_opts(self, s):
		ls = [el.strip() for el in s.split(';')]
		opts = {}
		for opt in ls:
			name, value = [el.strip() for el in opt.split('=')]
			opts[name] = value
		return opts

	def pack_file(self):
		self.path = askopenfilename(title='Pack a file', filetypes=(('all files', '.*'),))
		self.packfilebtn.destroy()
		self.packarcbtn.destroy()
		self.unpackbtn.destroy()
		self.filenameremind = Label(self, text=os.path.split(self.path)[-1])
		self.filenameremind.pack()
		self.outentry.pack()
		self.formatentry.pack()
		self.endianentry.pack()
		self.optentry.pack()
		self.confirm_callback = self.pack_file_final
		self.confirm.pack()
	
	def pack_file_final(self):
		path = self.path
		outname = self.outname.get()
		format = self.format.get().upper()
		endian = ENDIANS[self.endian.get().lower()]
		stringopts = self.opts.get()
		opts = self.parse_opts(stringopts)
		if format not in formats:
			showerror('Format error', 'This is not a supported format for packing')
			return
		pack(path, outname, format, endian, opts)

	def pack_archive(self):
		pass

	def unpack_file(self):
		pass

	def main(self, options):
		'''The plugin's entry point. Options are strings placed instead of the normal input name argument'''
		self.confirm_callback = self._null
		self.title = Label(self, text='3DSkit UI')
		self.title.pack()
		self.packarcbtn = Button(self, text='Pack an archive', command=self.pack_archive, width=50, height=6)
		self.packfilebtn = Button(self, text='Pack a file', command=self.pack_file, width=50, height=6)
		self.unpackbtn = Button(self, text='Unpack', command=self.unpack_file, width=50, height=6)
		self.packfilebtn.pack()
		self.packarcbtn.pack()
		self.unpackbtn.pack()
		self.outname = StringVar()
		self.outname.set('Output file name')
		self.outentry = Entry(self, textvariable=self.outname, width=50)
		self.format = StringVar()
		self.format.set('format')
		self.formatentry = Entry(self, textvariable=self.format, width=50)
		self.endian = StringVar()
		self.endian.set('Byte order (little for 3DS, big for WiiU)')
		self.endianentry = Entry(self, textvariable=self.endian, width=50)
		self.opts = StringVar()
		self.opts.set('Options in the format option1=value1; option2=value2; ...')
		self.optentry = Entry(self, textvariable=self.opts, width=50)
		self.confirm = Button(self, text='Confirm', command=self.confirm_callback)
		self.mainloop()

	def _null(self):
		pass

def main(options):
	window = _3DSkitUI(options)
