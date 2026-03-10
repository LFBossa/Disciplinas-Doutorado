%
% Data for final test CLINAL-II (November 2025)  
%addpath /home/usuario/regu/
close
clear;
close all
 % gera um sistema $Ax=b$ com $x$ image vetorizada.
 L = imread('lenaG.png');  L=double(L);
% imagesc(L);
% colormap gray; axis square
 xl = L(:); N=256; % solucao
 A = blur(N,5,3.5); 
 bl = A*xl;  bar= randn(size(bl)); bar = bar/norm(bar);
 bln = bl+0.05*bar*norm(bl); % Nivel de Ruido 5%
 figure, 
 subplot('Position',[0.3 0.55 0.4 0.4]),
 imagesc(reshape(xl,N,N)), colormap gray
 subplot('Position',[0.5 0.8 0.4 0.4]),
 imagesc(reshape(bln,N,N)), colormap gray
 title('Exata') 
 axis square
 axis off
 
 
 
[a,b,U,V] = GBK2(A,bln,20, true);
 
 norm(xl-bln)