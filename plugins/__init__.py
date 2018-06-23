# -*- coding:utf-8 -*-
import os
import sys
from util.filesystem import *
from util import error

plugin_path = None

def run_plugin(name, options, verbose):
	global plugin_path
	plugin_path = os.path.join(os.getcwd(), 'plugins', name, '')
	try:
		mod = __import__('plugins.%s.main' % name)
	except ImportError:
		error.PluginNotFoundError('Plugin %s is not found' % name)
	exec('mod.%s.main.main(options, verbose)' % name)

def getpath():
	global plugin_path
	return plugin_path

def readdata(name):
	global plugin_path
	return read(plugin_path + 'data' + os.path.sep + name)

def breaddata(name):
	global plugin_path
	return bread(plugin_path + 'data' + os.path.sep + name)
