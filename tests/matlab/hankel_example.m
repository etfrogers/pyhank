%%	Gaussian function
gauss1D = @(x, x0, FWHM) exp(-2*log(2)*((x-x0)/FWHM).^2);

%%	Initialise grid
disp('Initialising data ...');
Nr = 1024;			%	Number of sample points
r_max = .05;		%	Maximum radius (5cm)
dr = r_max/(Nr-1);	%	Radial spacing
nr = (0:Nr-1)';		%	Radial pixels
r = nr*dr;			%	Radial positions
Dr = 5e-3;			%	Beam radius (5mm)
Kr = 5000;			%	Propagation direction
Nz = 200;			%	Number of z positions
z_max = .25;		%	Maximum propagation distance
dz = z_max/(Nz-1);
z = (0:Nz-1)'*dz;	%	Popagation axis

%%	Setup Hankel transform structure
disp('Setting up Hankel transform structure ...');
H = hankel_matrix(0, r_max, Nr);
K = 2*pi*H.V;		%	Maximum K vector

%%	Generate electric field:
disp('Generating electric field ...');
Er = gauss1D(r, 0, Dr).*exp(i*Kr*r);	%	Initial field
ErH = spline(r, Er, H.r);				%	Resampled field

%%	Perform Hankel Transform
disp('Performing Hankel transform ...');
EkrH = qdht(ErH, H, 3);		%	Convert from physical field to physical wavevector

%%	Propagate beam
disp('Propagating beam ...');
EkrH_ = EkrH./H.JV;		%	Convert to scaled form for faster transform
phiz = @(z) (sqrt(K^2 - H.kr.^2) - K)*z;	%	Propagation phase
EkrHz = @(z) EkrH_ .* exp(i*phiz(z));		%	Apply propagation
ErHz = @(z) iqdht(EkrHz(z), H);				%	iQDHT (no scaling)
Erz = @(z) spline(H.r, ErHz(z).*H.JR, r);	%	Interpolate & scale output

Irz = zeros(Nr, Nz+1);
Irz(:, 1) = abs(Er).^2;
for n=1:Nz-1
	Irz(:, n+1) = abs(Erz(z(n+1))).^2;
end

disp('Complete!');