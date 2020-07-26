function fv = qdht(fr, s_HT, m)
%DHT: Quasi Discrete Hankel Transform
%
%	fv = qdht(fr, s_HT, m)
%
%	fv		= HT(fr) (sampled at s_HT.v)
%	fr		= Radial function (sampled at s_HT.r)
%	s_HT	= Hankel Transform structure (obtained using hankel_matrix)
%	m		= Mode (optional)
%
%	m = 0 (default):
%		fv = s_HT.T * fr;
%		fr & fv are the scaled functions (i.e. fr./s_HT.JR & fv./s_HT.JV)
%
%	m = 1:
%		fv = s_HT.T * (fr ./ s_HT.JR);
%		fv is the scaled function, fr the real function.
%
%	m = 2:
%		fv = s_HT.JV .* (s_HT.T * fr);
%		fv is the real function, fr is the scaled function
%
%	m = 3:
%		fv = s_HT.JV .* (s_HT.T * (fr ./ s_HT.JR));
%		fr, fv are the real distributions.
%
%	Need to call bessel_matrix(p, max_radius, N) once to obtain the structure s_HT.

if (~exist('m', 'var') || isempty(m))
	m = 0;
end

N2 = size(fr, 2);
if (N2>1)
	JR = s_HT.JR * ones(1, N2);
	JV = s_HT.JV * ones(1, N2);
else
	JR = s_HT.JR;
	JV = s_HT.JV;
end

switch (m)
	case (1)
		fv = s_HT.T * (fr ./ JR);
	case (2)
		fv = JV .* (s_HT.T * fr);
	case (3)
		fv = JV .* (s_HT.T * (fr ./ JR));
	otherwise
		fv = s_HT.T * fr;
end