import numpy as np

mTOcm = 100.                          # m to cm
cmTOm = 1. / 100.                     # cm to m
kmTOm = 1000.                         # km to m
mTOkm = 1. / 1000.                    # m to km
mTOpm = 1e12                          # m to pm
pmTOm = 1e-12                         # pm to m
mTOfm = 1e15                          # m to fm
fmTOm = 1e-15                         # fm to m
kgTOg = 1000.                         # kg to g
gTOkg = 1. / 1000.                    # g to kg
PaTOdynpcm2 = 10.                     # (*Pascals to dyne/cm^2*)
JouleTOerg = 1e7                      # Joule to erg
ergTOJoule = 1e-7                     # erg to Joule
inveVTOm = 197.3269788 * fmTOm        # (*MeV m*)
mTOinveV = 1. / (197.3269788 * fmTOm)  # (*(MeV m)^-1*)
inveVTOs = 6.582119514e-22
sTOinveV = 1. / (6.582119514e-22)
evTOkg = 1.78266191e-30              # MeV/c^2 to kg

# particle masses
mN = 939        # neutron mass (*MeV*)
mP = 938        # proton mass (*MeV*)
mE = 0.511      # electron mass (*MeV*)
mmu = 105.66    # muon mass (*MeV*)
mpion = 139.57061  # pion mass in MeV

# constants
cspeed = 299792458.      # speed of light m/s
hplanck = 6.62607015e-34  # plancks cosnst. J s
hbar = 6.582119569e-22   # reduced plancks const. MeV s
Msol = 1.989e30          # Mass of Sun in kg
Rsol = 696340000         # Solar radius in m
GNewt = 6.67430e-11      # Newtons constant SI
pi = np.pi                # Pi
kBMeV = 8.61733034e-11   # Boltzmann's const. in MeV/K
kB = 1.38064852e-23       # Boltzmann's const.
sigma_SB = 5.670374419e-8  #Stefan-Boltzman constant in SI units
munit = 1.66053906660e-27  # Atomic mass unit
alpha = 1./136.  # Fine structure constant
r0 = 1.2*fmTOm*mTOinveV  # r0 in inv MeV
lampi = 1/mpion # pion compton wavelength
echarge = 0.30286  # elementary charge in natural units

# TOV constants
M_scale = 1/(2*GNewt**(3/2)*np.sqrt(pi))
R_scale = 1/(2*np.sqrt(GNewt*pi))
