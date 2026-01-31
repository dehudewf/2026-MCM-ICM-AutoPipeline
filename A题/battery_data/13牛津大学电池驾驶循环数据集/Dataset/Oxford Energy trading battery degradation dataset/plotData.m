% Script to read and plot the data from the data from the dataset 
% 'Battery degradation data for energy trading with physical models'
% see 'Readme.txt' for a description of the data files.
%
% Copyright (c) 2020, The Chancellor, Masters and Scholars of the University 
% of Oxford, VITO nv, and the 'Battery degradation data for energy trading 
% with physical models' researchers. All rights reserved.
%
% THIS CODE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

clc
close all
clear

%% Optimal current profiles
% This section reads and plots the current profiles resulting from the
% three optimisations.

A = readmatrix('optimal_current_profiles.csv');
    % column 1 = time at the start of the step in the current profile in [s]
    % column 2 = current of the BMR profile [A], negative for charging
    % column 3 = current of the BMP profile [A], negative for charging
    % column 4 = current of the SPM profile [A], negative for charging
    
% plot the three optimal current profiles versus time
% We use a 'stairs' plot because current values are constant until the next
% step.
figure()
stairs(A(:,1)/3600,A(:,2))
hold on
stairs(A(:,1)/3600,A(:,3))
stairs(A(:,1)/3600,A(:,4))
legend('BMR','BMP','SPM')
ylabel('current [A]')
xlabel('time [hour]')
grid on
title('optimal current profiles')

%% Cell capacities
% This section plots the capacities of the six cells.
% Capacities were measured monthly.

% Read the files
fileNames = {'SPM_cell1_capacityData.csv','SPM_cell2_capacityData.csv',...
    'BMP_cell1_capacityData.csv','BMP_cell2_capacityData.csv',...
    'BMR_cell1_capacityData.csv','BMR_cell2_capacityData.csv'};
leg = {'SPM_1','SPM_2','BMP_1','BMP_2','BMR_1','BMR_2'};

cap = cell(6,1);
for i=1:6
    cap{i} = readmatrix(fileNames{i});
        % colum 1 gives the total time of the experiment in seconds
        % column 2 gives the (approximate) time spent while following the current profile
        %   in seconds
        % column 3 gives the cell capacity in [Ah]
end

% Plot the capacity versus 'profile time'.
% See 'Readme.txt' for an explanation about 'profile time'
% If you want to plot versus the total time, you need to use the first
% column of cap{i} instead of the second one.
figure()
for i=1:6
    plot(cap{i}(:,2)/3600,cap{i}(:,3),'.-')
    hold on
end
xlabel('time while following the current profile [hour]')
ylabel('capacity [Ah]')
legend(leg)
grid on

%% profile data
% This section plots the measured current, voltage and temperature while
% cells were following the current profile.

% Read the files
fileNames = {'SPM_cell1_profileData.csv','SPM_cell2_profileData.csv',...
    'BMP_cell1_profileData.csv','BMP_cell2_profileData.csv',...
    'BMR_cell1_profileData.csv','BMR_cell2_profileData.csv'};

profile = cell(6);
for i=1:6
    profile{i} = readmatrix(fileNames{i});
    % column 1: time [s] while following the current profile
    % column 2: current [A], negative when charging
    % column 3: voltage [V]
    % column 4: cell surface temperature in degrees celcius, 0 when no measurement available
    % column 5: environment temperature in degrees celcius, 0 when no measurement available
end

% Replace the '0' in the temperature measurement with NaN so it is not
% plotted
for i=1:6
    profile{i}((profile{i}(:,4)<=0),4) = NaN;
    profile{i}((profile{i}(:,5)<=0),5) = NaN;
end

% plot one figure per cell
% The first subplot shows the current and voltage (on the left and right
% y-axis). Note that we use the regular 'plot' command, although current
% values are constant per step in the profile. But because we are not 
% guaranteed a data point at the start or end of every step, we cannot use 
% the 'stairs' command to show the clear steps in the current profile.
% The second subplot shows the temperature of the cell and the environment.
for i=1:6
    figure()
    subplot(2,1,1)
        yyaxis left
        plot(profile{i}(:,1)/3600,profile{i}(:,2),'.-')
        ylabel('current [A]')
        yyaxis right
        plot(profile{i}(:,1)/3600,profile{i}(:,3),'.-')
        ylabel('voltage [V]')
        xlabel('time while following the current profile [hour]')
        grid on
        title(leg{i})
    subplot(2,1,2)
        yyaxis left
        plot(profile{i}(:,1)/3600,profile{i}(:,4),'.')
        ylabel('cell temperature [celcius]')
        yyaxis right
        plot(profile{i}(:,1)/3600,profile{i}(:,5),'.')
        ylabel('environment temperature [celcius]')
        xlabel('time while following the current profile [hour]')
        grid on
end



