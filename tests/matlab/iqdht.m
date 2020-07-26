function fr = iqdht(fv, s_HT, m)
%DHT: Quasi Discrete Hankel Transform
%
%	fr = iqdht(fv, s_HT, m)
%
%	fr		= HT(fv) (sampled at s_HT.v)
%	fr		= Radial function (sampled at s_HT.r)
%	s_HT	= Hankel Transform structure (obtained using hankel_matrix)
%	m		= Mode (optional)
%
%	m = 0 (default):
%		fr = s_HT.T * fv;
%		fr & fv are the scaled functions (i.e. fr./s_HT.JR & fv./s_HT.JV)
%
%	m = 1:
%		fr = (s_HT.T * fv) .* s_HT.JR;
%		fv is the scaled function, fr the real function.
%
%	m = 2:
%		fr = s_HT.T * (fv ./ s_HT.JV);
%		fv is the real function, fr the scaled function
%
%	m = 3:
%		fr = s_HT.JR .* (s_HT.T * (fv ./ s_HT.JV));
%		fr, fv are the real distributions.
%
%	Need to call bessel_matrix(p, max_radius, N) once to obtain the structure s_HT.

if (~exist('m', 'var') || isempty(m))
	m = 0;
end

N2 = size(fv, 2);
if (N2>1)
	JR = s_HT.JR * ones(1, N2);
	JV = s_HT.JV * ones(1, N2);
else
	JR = s_HT.JR;
	JV = s_HT.JV;
end

switch (m)
	case (1)
		fr = (s_HT.T * fv) .* JR;
	case (2)
		fr = s_HT.T * (fv ./ JV);
	case (3)
		fr = JR .* (s_HT.T * (fv ./ JV));
	otherwise
		fr = s_HT.T * fv;
end