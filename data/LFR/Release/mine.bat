@echo off

if not exist size50_mu0.1_minc25_maxc25 md size50_mu0.1_minc25_maxc25
cd size50_mu0.1_minc25_maxc25

..\binary_networks.exe -N 50 -k 3 -maxk 8 -mu 0.1 -minc 25 -maxc 25

cd ..\

if not exist size50_mu0.3_minc25_maxc25 md size50_mu0.3_minc25_maxc25
cd size50_mu0.3_minc25_maxc25

..\binary_networks.exe -N 50 -k 3 -maxk 8 -mu 0.3 -minc 25 -maxc 25

cd ..\

if not exist size50_mu0.5_minc25_maxc25 md size50_mu0.5_minc25_maxc25
cd size50_mu0.5_minc25_maxc25

..\binary_networks.exe -N 50 -k 3 -maxk 8 -mu 0.5 -minc 25 -maxc 25

cd ..\

if not exist size50_mu0.7_minc25_maxc25 md size50_mu0.7_minc25_maxc25
cd size50_mu0.7_minc25_maxc25

..\binary_networks.exe -N 50 -k 3 -maxk 8 -mu 0.7 -minc 25 -maxc 25

cd ..\

pause
