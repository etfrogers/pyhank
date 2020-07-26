function s_HT = hankel_matrix(p, max_radius, N)
%HANKEL_MATRIX: Generates data to use for Hankel Transforms
%
%	s_HT = hankel_matrix(p, max_radius, N)
%
%	s_HT	=	Structure containing data to use for the pQDHT
%	p		=	Transform order
%	max_radius		=	Radial extent of transform
%	N		=	Number of sample points
%
%	s_HT:
%		p, max_radius, N		=	As above
%		alpha		=	Roots of the pth order Bessel fn.
%					=	[alpha_1, alpha_2, ... alpha_N]
%		alpha_N1	=	(N+1)th root
%		r			=	Radial co-ordinate vector
%		v			=	frequency co-ordinate vector
%		kr			=	Radial wave number co-ordinate vector
%		V			=	Limiting frequency
%					=	alpha_N1/(2piR)
%		S			=	RV product
%					=	2*pi*max_radius*V
%		T			=	Transform matrix
%		JR, JV		=	Transform vectors
%					=	J_p1(alpha)/max_radius  or Jp1(alpha)/V where p1 = p+1
%
%	The algorithm used is that from:
%		"Computation of quasi-discrete Hankel transforms of the integer
%		order for propagating optical wave fields"
%		Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
%		J. Opt. Soc. Am. A 21(1) 53-58 (2004)
%
%	The algorithm also calls the function:
%	zn = bessel_zeros(1, p, N+1, 1e-6),
%	where p and N are defined above, to calculate the roots of the bessel
%	function. This algorithm is taken from:
%  		"An Algorithm with ALGOL 60 Program for the Computation of the
%  		zeros of the Ordinary Bessel Functions and those of their
%  		Derivatives".
%  		N. M. Temme
%  		Journal of Computational Physics, 32, 270-279 (1979)

s_HT.p = p;
s_HT.max_radius = max_radius;
s_HT.N = N;

%	Calculate N+1 roots:
alpha = bessel_zeros(1, s_HT.p, s_HT.N+1, 1e-6);
s_HT.alpha = alpha(1:end-1);
s_HT.alpha_N1 = alpha(end);

%	Calculate co-ordinate vectors
s_HT.r = s_HT.alpha * s_HT.max_radius / s_HT.alpha_N1;
s_HT.v = s_HT.alpha / (2*pi * s_HT.max_radius);
s_HT.kr = 2*pi * s_HT.v;
s_HT.V = s_HT.alpha_N1 / (2*pi * s_HT.max_radius);
s_HT.S = s_HT.alpha_N1;

%	Calculate hankel matrix and vectors
Jp = besselj(p, (s_HT.alpha) * (s_HT.alpha') / s_HT.S);
Jp1 = abs(besselj(p+1, s_HT.alpha));
s_HT.T = 2*Jp./(Jp1 * Jp1' * s_HT.S);
s_HT.JR = Jp1 / s_HT.max_radius;
s_HT.JV = Jp1 / s_HT.V;