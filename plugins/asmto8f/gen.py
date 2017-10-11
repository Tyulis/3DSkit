# -*- coding:utf-8 -*-
import struct
from plugins import readdata
from util.rawutil import TypeReader
from util.funcops import FakeFile, toascii
from util import error

#table bases on PRAMA Initiative's one
ITEM_TABLE = (
	'Flash', 'Master Ball', 'Hyper Ball',
	'Super Ball', 'Poké Ball', 'Carte',
	'Bicyclette', '?????', 'Safari Ball',
	'Pokédex', 'Pierre Lune', 'Antidote',
	'Anti-Brûle', 'Antigel', 'Réveil',
	'Anti-Para', 'Guérison', 'Potion Max',
	'Hyper Potion', 'Super Potion', 'Potion',
	'Badge Roche', 'Badge Cascade', 'Badge Foudre',
	'Badge Prisme', 'Badge Âme', 'Badge Marais',
	'Badge Volcan', 'Badge Terre', 'Corde Sortie',
	'Repousse', 'Vieil Ambre', 'Pierre Feu', 'Pierre Foudre', 'Pierre Eau', 'PV Plus',
	'Protéine', 'Fer', 'Carbone', 'Calcium',
	'Super Bonbon', 'Fossile Dôme', 'Nautile',
	'Clé Secrète', '?????', 'Bon Commande',
	'Précision +', 'Pierre Plante', 'Carte Magn.',
	'Pépite', 'PP Plus (non functional)',
	'Poképoupée', 'Total Soin', 'Rappel',
	'Rappel Max', 'Défense Spec', 'Super Repousse',
	'Max Repousse', 'Muscle +', 'Jetons',
	'Eau Fraîche', 'Soda Cool', 'Limonade',
	'Passe Bateau', 'Dents d\'or', 'Attaque +',
	'Défense +', 'Vitesse +', 'Spécial +',
	'Boite Jetons', 'Colis Chen', 'Cherch\'Objet',
	'Scope Sylph', 'Pokéflute', 'Clé Asc.',
	'Multi Exp.', 'Canne', 'Super Canne',
	'Méga Canne', 'PP Plus', 'Huile', 'Huile Max',
	'Élixir', 'Max Élixir', '2ème SS', '1er SS',
	'RDC', '1er Étage', '2ème Étage', '3ème Étage',
	'4ème Étage', '5ème Étage', '6ème Étage',
	'7ème Étage','8ème Étage', '9ème Étage',
	'10ème Étage', '4ème SS', 'w üm\'||lm||',
	'ws*l\'||lm||', 'v Aft||lm||', 'ûc\' è||lm||',
	'êu\'c\'m\'||lm||', 'üwj\'é||lm||',
	'||lf||lm||', 'êôA ||lm||', '\\-g*||lm||',
	'A /', 'êj\'à', '*i l *', 'Lg|||-', '\\-g*',
	'?QGn?Sl', 'Gn?Sl', '?Q;MP-', ';MP-',
	'DHNh l T4', '*ö****j\'*', '_/*-', '4',
	'*4ô ê ü*?', '*8\\û', '8*û-', '4û hA *',
	'89*****l\'êGp*|||', '<RIVAL>* *A***** *ôp**',
	'<RIVAL>********', '......* *||| ** ; *',
	'o*', '**ASPICOT/', '4/î*4\\îyü. ... ... 4*',
	'4*î*', 'K***... ...*P*|||î a',
	'ECHANGE TERMINÉ', 'Dommage! L\'échange',
	'ù', '| e<RIVAL>* \'*||*',
	'****PkMn***ö***ASPICOTö', '*SG*', '*HG*',
	'**l\'êo qB** ......*', 'CENTRE TROC',
	'p’ à**ö/\** |||*METAMORPH', '*aä/*** |||*ö/',
	'8 \\', 'ANIMATION COMBAT', 'STYLE COMBAT',
	'RETOUR', '*?B4*', '\*/*2p*', '\'*',
	'**H***PkMnH*', '*+H*', '**I*', '**I*',
	'* D//*\'*** ......*', '8', 'APOKEDRES. * : *',
	'** p* ***C ?', '8', '\\**à **', 'n*',
	'p** ***Q I3*4* h', '**', '*Q n ô4* hâ ov*',
	'ô4*î8/â4*î8*ûpH*****', 'ABCDEFGHIJKLMNOPQRST',
	'ov*** * ä***ö** a*',
	'<YOU>||* ?ä4C 8*********', 'm*',
	'â **2*uä4C *c’vh***y’v', 'NOM DU RIVAL?',
	'NOM?', 'SURNOM?', 'ps*?L \L4/îh\**KL *', '8',
	'\? *||| , ****/**D**sä',
	'ps*ASPICOTL \\L4/îh\\***L *', '8',
	'\\* *||| ,**', 'd’*aä*** ö|||** ……* * : *', 'NOM:', 'NOM:', '**', '*5*z\**|||.CL*:',
	'BLUE', 'REGIS', 'JEAN', 'NOM :', 'RED',
	'SACHA', 'PAUL', '<NOTHING>', '|', '*||M\\',
	'**M\\', 'CS01', 'CS02', 'CS03', 'CS04',
	'CS05', 'CT01', 'CT02', 'CT03', 'CT04', 'CT05',
	'CT06', 'CT07', 'CT08', 'CT09', 'CT10', 'CT11',
	'CT12', 'CT13', 'CT14', 'CT15', 'CT16', 'CT17',
	'CT18', 'CT19', 'CT20', 'CT21', 'CT22', 'CT23',
	'CT24', 'CT25', 'CT26', 'CT27', 'CT28', 'CT29',
	'CT30', 'CT31', 'CT32', 'CT33', 'CT34', 'CT35',
	'CT36', 'CT37', 'CT38', 'CT39', 'CT40', 'CT41',
	'CT42', 'CT43', 'CT44', 'CT45', 'CT46', 'CT47',
	'CT48', 'CT49', 'CT50', 'CT51', 'CT52', 'CT53',
	'CT54', '<end (return button)>'
)

POKE_TABLE = (
	'\'M', 'Rhinoféros', 'Kangourex',
	'Nidoran (mâle)', 'Mélofée', 'Piafabec',
	'Voltorbe', 'Nidoking', 'Flagadoss',
	'Herbizarre', 'Noadkoko', 'Excelangue',
	'Noeunœuf', 'Tadmorv', 'Ectoplasma',
	'Nidoran (femelle)', 'Nidoqueen', 'Osselait',
	'Rhinocorne', 'Lokhlass', 'Arcanin', 'Mew',
	'Léviator', 'Kokiyas', 'Tentacool',
	'Fantominus', 'Insécateur', 'Stari', 'Tortank',
	'Scarabrute', 'Saquedeneu', 'MissingNo.', 'MissingNo.', 'Caninos', 'Onix', 'Rapasdepic',
	'Roucool', 'Ramoloss', 'Kadabra', 'Gravalanch',
	'Leveinard', 'Machopeur', 'M.Mime', 'Kicklee',
	'Tygnon', 'Arbok', 'Parasect', 'Psykokwak',
	'Soporifik', 'Grolem', 'MissingNo.', 'Magmar',
	'MissingNo.', 'Elektek', 'Magnéton', 'Smogo',
	'MissingNo.', 'Férosinge', 'Otaria',
	'Taupiqueur', 'Tauros', 'MissingNo.',
	'MissingNo.', 'MissingNo.', 'Canarticho',
	'Mimitoss', 'Dracolosse', 'MissingNo.',
	'MissingNo.', 'MissingNo.', 'Doduo', 'Ptitard',
	'Lippoutou', 'Sulfura', 'Artikodin',
	'Electhor', 'Métamorph', 'Miaouss', 'Krabby',
	'MissingNo.', 'MissingNo.', 'MissingNo',
	'Goupix', 'Feunard', 'Pikachu', 'Raichu',
	'MissingNo.', 'MissingNo.', 'Minidraco',
	'Draco', 'Kabuto', 'Kabutops', 'Hypotrempe',
	'Hypocéan', 'MissingNo.', 'MissingNo.',
	'Sabelette', 'Sablaireau', 'Amonita',
	'Amonistar', 'Rondoudou', 'Grodoudou',
	'Evoli', 'Voltali', 'Pyroli', 'Aquali',
	'Machoc', 'Nosférafti', 'Abo', 'Paras',
	'Têtarte', 'Tartard', 'Aspicot', 'Coconfort',
	'Dardargnan', 'MissingNo.', 'Dodrio',
	'Colossinge', 'Triopikeur', 'Aéromite',
	'Lamantine', 'MissingNo.', 'MissingNo.',
	'Chenipan', 'Chrysacier', 'Papilusion',
	'Mackogneur', 'MissingNo.', 'Akwakwak',
	'Hypnomade', 'Nosféralto', 'Mewtwo', 'Ronflex',
	'Magicarpe', 'MissingNo.', 'MissingNo.',
	'Grotadmorv', 'MissingNo.', 'Kraboss',
	'Crustabri', 'MissingNo.', 'Électrode',
	'Mélodelfe', 'Smogogo', 'Persian', 'Ossatueur',
	'MissingNo.', 'Spectrum', 'Abra', 'Alakazam',
	'Roucoups', 'Roucarnage', 'Staross',
	'Bulbizarre', 'Florizarre', 'Tentacruel',
	'MissingNo.', 'Poissirène', 'Poissoroy',
	'MissingNo.', 'MissingNo.', 'MissingNo.',
	'MissingNo.', 'Ponyta', 'Galopa', 'Rattata',
	'Rattatac', 'Nidorino', 'Nidorina',
	'Racaillou', 'Porygon', 'Ptéra', 'MissingNo.',
	'Magnéti', 'MissingNo.', 'MissingNo.',
	'Salamèche', 'Carapuce', 'Carabaffe', 'Dracaufeu', 'MissingNo.', 'MissingNo.',
	'MissingNo.', 'MissingNo.', 'Mystherbe',
	'Ortide', 'Rafflésia', 'Chétiflor',
	'Boustiflor', 'Empiflor', '* * ;', 'ù ö',
	'ö ,', ',4 î', 'h û', '/ ä', 'Pk Mn A',
	',Q ||, R 4 î', '♀P, î, î Y', ',Z 6 ö',
	'Gamin', 'Scout', 'Fillette', 'Marin',
	'Dresseur Jr Mâle', 'Dresseur Jr Femelle',
	'Pokémaniac', 'Intello', 'Montagnard',
	'Motard', 'Pillard', 'Mécano', 'Jongleur',
	'Pêcheur', 'Nageur', 'Loubard', 'Croupier',
	'Canon', 'Kinésiste', 'Rocker', 'Jongleur',
	'Dompteur', 'Ornithologue', 'Karatéka',
	'Rival', 'Prof. Chen', 'Chef', 'Scientifique',
	'Giovanni', 'Rocket', 'Topdresseur mâle',
	'Topdresseur femelle', 'Aldo', 'Pierre',
	'Ondine', 'Major Bob', 'Erika', 'Koga',
	'Auguste', 'Morgane', 'Gentleman', 'Rival',
	'Rival', 'Olga', 'Exorciste', 'Agatha',
	'Peter', 'Pk?-', '*1 ,', 'ô 4 8', 'U ä 4',
	'4CT âä', '*,', 'ää +', '......p\' à'
)
	

def generate(bin, lang):
	if (len(bin) % 2) != 0:
		bin += b'\x01'
	lst = '<anything>\n7ème étage\n'
	for i in range(0, len(bin), 2):
		itemid = bin[i]
		itemnum = bin[i + 1]
		item = ITEM_TABLE[itemid]
		ln = '%s ($%02x) x%d\n' % (item, itemid, itemnum)
		lst += ln
	return lst

def gensetup(bin):
	ptr = 0
	tp = TypeReader()
	tp.byteorder = '<'
	print('Number of pokes: 5')
	f = FakeFile(bin, '<')
	try:
		pk1 = f.uint8()
		print('Poké nº1: %s ($%02x)' % (POKE_TABLE[pk1], pk1))
		pk2 = f.uint8()
		print('Poké nº2: %s ($%02x)' % (POKE_TABLE[pk2], pk2))
		pk3 = f.uint8()
		print('Poké nº3: %s ($%02x)' % (POKE_TABLE[pk3], pk3))
		pk4 = f.uint8()
		print('Poké nº4: %s ($%02x)' % (POKE_TABLE[pk4], pk4))
		pk5 = f.uint8()
		print('Poké nº5: %s ($%02x)' % (POKE_TABLE[pk5], pk5))
		print('')
		pv1 = 0 
		pv2 = f.uint8()
		pv = (pv1 << 8) + pv2
		print('Poké nº1 PV: %d ($%04x)' % (pv, pv))
			
	except IOError:
		pass  #end of code
	except struct.error:
		pass  #idem

def disinventory(inv):
	l = inv.splitlines()
	code = []
	items = [toascii(el.replace(' ', '').lower().replace('-', '')) for el in ITEM_TABLE]
	for i, ln in enumerate(l):
		if ln == '':
			break
		item = ln.split()
		name = toascii(''.join(item[0:-1]).lower().replace('-', ''))
		quantity = int(item[-1].strip('x'))
		try:
			code.append(items.index(name))
		except ValueError:
			print('Unknown item %s at line %d' % (' '.join(item[0:-1]), i))
			return b''
		if quantity > 255:
			print('Too much items %s (%d) at line %d' % (''.join(item[:-1]), quantity, i))
			return b''
		code.append(quantity)
	return bytes(code)
