TITLE Calcium ion accumulation and buffering
:
: This is the final decided version of a simplified Calcium dynamics adapted from the 4 shell Migliore et al. 1995 model
: This is formally equivalent to strategy 15, but with codes streamlined.
:
: It has one shell, which is defined as the region between the cell membrane and 1/6 diameter below cell membrane
:
: cai is defined as Calcium concentration within this shell
:
: This model has the following mechanisms affecting cai
: - an equation that converts the ica into calcium influx to the shell
: - calcium buffering within the shell
: - a calcium pump that moves calcium within the shell to extracellular space

NEURON {
	THREADSAFE
	SUFFIX cad
	USEION ca READ cao, cai, ica WRITE cai, ica
	RANGE ipump
}

UNITS {
    (mol)   = (1)
	(molar) = (1/liter)			: moles do not appear in units
	(mM)	= (millimolar)
	(um)	= (micron)
	(mA)	= (milliamp)
	PI	= (pi) (1)
	FARADAY = (faraday) (10000 coulomb)
}

PARAMETER {
	DFree = .6	(um2/ms)
	diam		(um)
	cao		(mM)
    area        (um2)
    k1buf = 500	(/mM-ms)
	k2buf = 0.5	(/ms)
        k1=1.e10            (um3/s)
        k2=50.e7            (/s)	: k1*50.e-3
        k3=1.e10            (/s)	: k1
        k4=5.e6	            (um3/s)	: k1*5.e-4
}

STATE {
	ca_i    (mM) <1e-5>
    CaBuffer_shell  (mM)
	Buffer_shell    (mM)
        pump            (mol/cm2) <1.e-3>
        pumpca          (mol/cm2) <1.e-15>
}

LOCAL totpump, kd, totbuf

INITIAL {
	ca_i = cai
    last_ipump = 0
    totbuf=1.2 (mM)
           kd=k2buf/k1buf
    CaBuffer_shell =(totbuf*cai)/(kd+cai)
    Buffer_shell = totbuf - CaBuffer_shell

    totpump=0.2 (mol/cm2)
       pump=totpump/(1+(1.e-18)*k4*cao/k3)
       pumpca=2.e-22

    vol = diam * diam * PI * (1/2)^2
	shell_vol = (11/36) * vol
}

ASSIGNED {
	last_ipump      (mA/cm2)
	cai     (mM)
	ipump (mA/cm2)
	ica		(mA/cm2)
	vol    (um2)
	shell_vol (um2)
}


BREAKPOINT {
	SOLVE state METHOD sparse
	last_ipump = ipump
	ica = ipump
}

KINETIC state {
	COMPARTMENT shell_vol*1(um) {ca_i CaBuffer_shell Buffer_shell}
        COMPARTMENT (1.e10)*area {pump pumpca}
        COMPARTMENT (1.e15)*1 (liter) {cao}

    ~ ca_i << (-(ica-last_ipump)*PI*diam*1(um)/(2*FARADAY))

	~ ca_i + Buffer_shell <-> CaBuffer_shell (k1buf*shell_vol*1(um), k2buf*shell_vol*1(um))

    ~ca_i + pump <-> pumpca ((1.e-11)*k1*area, (1.e7)*k2*area)
    ~pumpca       <-> pump + cao ((1.e7)*k3*area, (1.e-11)*k4*area)

	ipump = 2*FARADAY*(f_flux-b_flux)/area
	cai = ca_i
}