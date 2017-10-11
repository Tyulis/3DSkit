# -*- coding:utf-8 -*-
from collections import OrderedDict
from util.funcops import ClsFunc

class dump (ClsFunc):
	def main(self,tree,customs=[]):
		self.customs=customs
		return self.dumpNode(tree)
	
	def dumpNode(self,node):
		final=''
		for key in node.keys():
			if key.__class__==str:
				if key.startswith('__'):
					continue
			if node[key].__class__ in [dict,OrderedDict]+self.customs:
				blk=self.dumpNode(node[key])
				blk=self.indent(blk)
				final+='%s: \n'%repr(key)
				final+=blk
			elif node[key].__class__ in (list,tuple):
				dic=dict(enumerate(node[key]))
				blk=self.dumpNode(dic)
				blk=self.indent(blk)
				final+='%s: %s\n'%(repr(key),str(node[key].__class__.__qualname__))
				final+=blk
			else:
				final+='%s: %s\n'%(repr(key),repr(node[key]))
		return final
	
	def indent(self,s):
		ret=''
		for line in s.splitlines():
			ret+='	|%s\n'%line
		return ret

class load (ClsFunc):
	def main(self,data):
		return self.loadNode(data)
	
	def loadNode(self,node):
		dic=OrderedDict()
		node=node.splitlines()
		i=0
		while True:
			try:
				line=node[i].split(': ')
			except IndexError:
				break
			if line[1].strip() in ('','list','tuple'):
				subnode=''
				for subline in node[i+1:]:
					if subline.startswith('\t|'):
						subnode+=subline+'\n'
						i+=1
					else:
						break
				res=self.loadNode(self.unindent(subnode))
				if line[1]=='list':
					res=list(res.values())
				elif line[1]=='tuple':
					res=tuple(res.values())
				dic[eval(line[0])]=res
			else:
				if line[1] in ('true','false','none'):
					line[1]=line[1].capitalize()
				dic[eval(line[0])]=eval(line[1])
			i+=1
		return dic
	
	def unindent(self,s):
		s=s.splitlines()
		ret=''
		for line in s:
			ret+='%s\n'%line[2:]
		return ret
