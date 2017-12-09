# -*- coding: utf-8 -*-
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from plugins import *

def main(options, verbose):
	'''The plugin's entry point. Options are strings placed instead of the normal input name argument'''
	print('Hello 3DSkit!')
