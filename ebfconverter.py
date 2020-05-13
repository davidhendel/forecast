import sys
import astropy.table as table
import ebf

t =  table.Table()
f = ebf.read(sys.argv[1])
glat, glon, rad, smass, age, feh = [f[_] for _ in ['glat', 'glon', 'rad', 'smass', 'age', 'feh']]
for col in ['glat', 'glon', 'rad', 'smass', 'age', 'feh']:
    t.add_column(table.Column(eval(col),col))
t.write(sys.argv[1].split('.ebf')[0]+'.fits',overwrite=True)