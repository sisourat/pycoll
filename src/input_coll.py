 # input
debug = False
analyze = False
nodiag = False
nstep_analysis = 10

orb =  "HF"  # HF or modpot or modpot_erf
ne = 2

tbasis = {'H': 'aug-cc-pvdz' }
tgeom = "H 0 0 0.0 ; H 0 0 1.401 ; "
isotop1 = 1.0
isotop2 = 1.0 # If D or T instead of H
talp = []  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
tcoef = []
tcenter = [[0,0,0]]
tpower = []
tcharge = 0
tspin = 0

pbasis = {'H@2': 'aug-cc-pvdz'}
elp = "H@2"
xp = 0
yp = 0
zp = -1000.0
palp = []
pcoef = []
ppower = []
pcenter = [[0,0,zp]]
pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
pcharge = 0
pspin = 1

i_init = 0
dtime = 0.05

zmax = 60.0
ngrid = 100
gridtype = 'exp'  #lin or exp
vproj = 0.01
bmin =  0.5
bmax =  6.5
nbb = 12

xmlfile = 'csfs.xml'
# sta index: alp, re, de, nvib
sta_list= {25:(1.04, 1.401, 0.1744 ,5), 24:(0.72, 2.0, 0.1026, 6), 22:(0.72, 2.0, 0.1026, 7), 18:(0.72, 2.0, 0.1026, 4), 19:(0.72, 2.0, 0.1026, 4), 20:(0.72, 2.0, 0.1026, 4) }
