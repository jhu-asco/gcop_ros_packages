

% file='/home/subhransu/gcop_ros_est/calib/mavros_mag_calib.dat';
file='/home/subhransu/gcop_ros_est/calib/mavros_mag_noise.dat';
fid = fopen(file,'r');fgetl(fid);
data = textscan(fid,'%*u64 %*u32 %*u64 %*s %f64 %f64 %f64 %*f %*f %*f %*f %*f %*f %*f %*f %*f','Delimiter',',');
fclose(fid);
mag_raw = cell2mat(data);
clear data;

R = [0 1 0; -1 0 0; 0 0 1];
mag = mag_raw*R';
mag_h = [mag_raw,ones([size(mag_raw,1),1]) ];
% 
% transform =1.0e+04 *[-0.062949872640317   2.273324797353632   0.001240242291801
%                      -2.154178589047590   0.062949872640317   0.034343121079356
%                      -0.034343121079356   0.001240242291801   2.468964306660896]/R;
% center = 1.0e-04 *[ 0.12763462704150, -0.095049461565055, -0.133981489367257]';
% 
% centers = repmat(center, 1, size(mag,1));
% transformed_mag = transform*(mag' - centers);

magcal_linear=      [-0.062949872640317e4 , 2.273324797353632e4 , 0.001240242291801e4
                     -2.154178589047590e4 , 0.062949872640317e4 , 0.034343121079356e4
                     -0.034343121079356e4 , 0.001240242291801e4 , 2.468964306660896e4];
                  
magcal_translation= [-0.284005441644962   , 0.201320273998919   , 0.333901512305726] ; 

T_new = [affine_linear,affine_translation;[0 0 0 1]];
mag_trfmd = mag_h*T_new';
mag_trfmd(:,4)=[];

mag_cov = cov(mag_raw)
size(mag_cov)

norm(transform*(mag(10,:)' - center))
