% test ellipsoid fit

clear all;
close all;

file='/home/subhransu/gcop_ros_est/calib/mavros_mag_calib.dat';
R = [1 0 0; 
    0 1 0;
    0 0 1];

% file='/home/subhransu/gcop_ros_est/calib/imu_3dm_mag_calib.dat';
% R = [1 0 0; 0 1 0; 0 0 1];

fid = fopen(file,'r');fgetl(fid);
data = textscan(fid,'%*u64 %*u32 %*u64 %*s %f64 %f64 %f64 %*f %*f %*f %*f %*f %*f %*f %*f %*f','Delimiter',',');
fclose(fid);
mag_raw_all = cell2mat(data);
clear data;


keep = 1:5:length(mag_raw_all);%use one in 5 data
mag = mag_raw_all(keep,:)*R';
clear mag_raw_all;


mx = mag(:,1);
my = mag(:,2);
mz = mag(:,3);


% do the fitting
[ center, radii, evecs, v ] = ellipsoid_fit( [mx my mz ] );
fprintf( 'Ellipsoid center: %.3g %.3g %.3g\n', center );
fprintf( 'Ellipsoid radii : %.3g %.3g %.3g\n', radii );
fprintf( 'Ellipsoid evecs :\n' );
fprintf( '%.3g %.3g %.3g\n%.3g %.3g %.3g\n%.3g %.3g %.3g\n', ...
    evecs(1), evecs(2), evecs(3), evecs(4), evecs(5), evecs(6), evecs(7), evecs(8), evecs(9) );
fprintf( 'Algebraic form  :\n' );
fprintf( '%.3g ', v );
fprintf( '\n' );

% draw data
plot3( mx, my, mz, '.r' );
hold on;

shiftx = -0.000;
shifty = -0.000;
shiftz = -0.000;
maxd = 2*max( radii );
step = maxd / 50;
[ x, y, z ] = meshgrid( -maxd:step:maxd + shiftx, -maxd:step:maxd + shifty, -maxd:step:maxd + shiftz );

Ellipsoid = v(1) *x.*x +   v(2) * y.*y + v(3) * z.*z + ...
          2*v(4) *x.*y + 2*v(5)*x.*z + 2*v(6) * y.*z + ...
          2*v(7) *x    + 2*v(8)*y    + 2*v(9) * z;
p = patch( isosurface( x, y, z, Ellipsoid, 1 ) );
set( p, 'FaceColor', 'g', 'EdgeColor', 'none' );
view( -70, 40 );
axis vis3d;
camlight;
lighting phong;
hold off;

% % transformed data
scale = eye(3,3);
scale(1,1) = radii(1);
scale(2,2) = radii(2);
scale(3,3) = radii(3);
transform = evecs*inv(scale)*evecs';
centers = repmat(center, 1, size(mag,1));
transformed_mag = transform*(mag' - centers);
figure, plot3(transformed_mag(1,:),transformed_mag(2,:),transformed_mag(3,:)) 
[ centert, radiit, evecst, vt ] = ellipsoid_fit( transformed_mag' );

% final result
affine_linear = transform*R
affine_translation = -transform*center;
affine_translation'
T = [affine_linear,affine_translation; 0 0 0 1]
