******************************************************************************************
Battery degradation data for energy trading with physical models
******************************************************************************************
'Battery degradation data for energy trading with physical models' contains measurements of battery ageing data from 6 lithium-ion cells following current profiles
corresponding to optimal trading strategies for stationary batteries in the Belgian day-ahead market of 2014.
A full explanation is given in the referenes given below, as well as in the PhD thesis 'Degradation-aware optimal control of grid-connected lithium-ion batteries', Jorn Reniers, University of Oxford, 2019.

There are three sets of csv sheets and one Matlab script to read them.
1. optimal current profiles
2. measured cycling data when cells were following these current profiles
3. monthly capacity measurements while cells were following the current profiles.
Further details are provided below.

---- PLEASE READ THE FOLLOWING MESSAGE CAREFULLY BEFORE PROCEEDING FURTHER:
If you make use of our data, please cite our dataset directly using its DOI, as well as one of the following papers:
Improving optimal control of grid-connected lithium-ion batteries through more accurate battery and degradation modelling, Journal of Power Sources 379 (2018) 91–102, DOI: https://doi.org/10.1016/j.jpowsour.2018.01.004.
Degradation-constrained control of batteries for energy trading using physical models, under review.

Thank you for your interest in our work.
David Howey 
Grietus Mulder
Jorn Reniers
May 2020

******************************************************************************************
Description of data contained in 'optimal_current_profiles.csv'
******************************************************************************************

This file contains three current profiles.
The first column indicates the time at the start of each step in the profile in seconds. The duration of every step is 15 minutes, or 900 seconds.
The second column is the current profile from the optimisation using the revenue-maximising bucket model in Amperes. Negative values indicate charging.
The third column is the current profile from the optimisation using the profit-maximising bucket model in Amperes. Negative values indicate charging.
The fourth column is the current profile from the optimisation using the profit-maximising single particle model in Amperes. Negative values indicate charging.

For information about the models and optimisation, the reader is referred to 'Improving optimal control of grid-connected lithium-ion batteries through more accurate battery and degradation modelling'.
Simulations by: Jorn Reniers, jorn.reniers@eng.ox.ac.uk
Supervisors: David A. Howey, david.howey@eng.ox.ac.uk and Grietus Mulder, grietus.mulder@vito.be
website: http://howey.eng.ox.ac.uk/


******************************************************************************************
Description of data contained in 'abc_cellx_profileData.csv'
******************************************************************************************
where abs is one of the following: BMP, BMR, SPM; and where x is either 1 or 2.

Recorded by: Jorn Reniers, jorn.reniers@eng.ox.ac.uk and Grietus Mulder, grietus.mulder@vito.be
Supervisor: David A. Howey, david.howey@eng.ox.ac.uk
Place of tests: EnergyVille, Belgium
Website: https://www.energyville.be/en
Test subjects: 6 x Kokam CO LTD, SLPB78205130H, 16 Ah.
Battery tester: PEC SBT8050
Environment: no thermal chamber, the cells were resting on a shelf in a room. The temperature of the room was controlled by the building AC system, and is added to the spreadsheets with the data.

These files contain the measurements of the six cells while they were being cycled with the aforementioned current profiles.
The first three letters indicate which current profile the cell was following (BMR = revenue-maximising bucket model; BMP = profit-maximising bucket model; SPM = single particle model).
Two cells followed each profile, differentiated by the digit in the document name.

The first column gives the 'profile time' in seconds. This time only accounts for the periods the cells were being cycled with the current from the profiles; time for check-up cycles and down-time have been omitted.
The second column gives the current in Amperes, where negative values indicate the cell is charging.
The third column gives the cell voltage.
The fourth column gives the surface temperature of the cell in centigrade, measured by a thermocouple located at the estimated hot-spot of the cells. A value of '0' indicate no measurement was available.
The fifth column gives the environmental temperature in centigrade. A value of '0' indicate no measurement was available.

While effort was done to align these measurements with the original current profiles by removing time spent on check-up cycles, down time, etc., they do not match exactly.
For instance, there was no guaranteed data point logged when a new current was applied, or when the voltage had changed by a certain amount. Therefore, it was not possible to exactly match the measurements here with the current profiles from the previous file.
Therefore, when the user needs the current profiles, it is best to use those from 'optimal_current_profiles.csv'.

The battery tester was programmed to cycle the cells with the current profiles while respecting the voltage limits.
If a voltage limit was reached during a step of the profile, the voltage was held constant for the remaining time of this step.
For instance, if the profile instructed the cell to charge at 5A for 15 minutes, but the maximum voltage was reached after 12 minutes, then for 3 minutes a CV phase was done at the maximum voltage.
This guaranteed that every step took exactly 15 minutes, and that the cells followed the current profile as good as possible.
For the revenue-maximising bucket model (BMR) and the profit-maximising single particle model, the voltage limits were set to 2.7 V and 4.2 V (the full range of the cell).
For the profit-maximising bucket model (BMP), the voltage limits were set to 3.42 V and 4.08 V, corresponding to 10 % and 90 % state of charge.

******************************************************************************************
Description of data contained in 'abc_cellx_capacityData.csv'
******************************************************************************************
where abs is one of the following: BMP, BMR, SPM; and where x is either 1 or 2.

Recorded by: Jorn Reniers, jorn.reniers@eng.ox.ac.uk and Grietus Mulder, grietus.mulder@vito.be
Supervisor: David A. Howey, david.howey@eng.ox.ac.uk
Place of tests: EnergyVille, Belgium
Website: https://www.energyville.be/en
Test subjects: 6 x Kokam CO LTD, SLPB78205130H, 16 Ah.
Battery tester: PEC SBT8050
Environment: no thermal chamber, the cells were resting on a shelf in a room. The temperature of the room was controlled by the building AC system.

These files contain the capacity measurements of the six cells during the test. 
The first three letters indicate which current profile the cell was following (BMR = revenue-maximising bucket model; BMP = profit-maximising bucket model; SPM = single particle model).
Two cells followed each profile, differentiated by the digit in the document name.

The first column gives the total time the experiment has been running when the check-up cycle starts in seconds.
The second column gives the 'profile time', which omits the time for the check-up itself, as well as down time, etc. Therefore, this time allows users to see where in the current profile from the other documents this check-up is performed.
The third column gives the measured capacity in Ampere-hours.

The capacity was measured by a CCCV charge (1C CC until 4.2 V, CV at 4.2 V until a limit current of 0.01 C or 0.16 A), and a CCCV discharge (1C until 2.7 V, CV at 2.7 V until a limit current of 0.01 C or 0.16 A).
The cell was relaxed for one hour between subsequent charges and discharges.
Two full cycles were done (i.e. two charges and two discharges), and the capacity is the mean of the four measured values.

******************************************************************************************
These data are copyright (c) 2020, The Chancellor, Masters and Scholars of the University of Oxford, VITO nv, and the 'Battery degradation data for energy trading with physical models' researchers. All rights reserved.

This 'Battery degradation data for energy trading with physical models' is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual contents of the database are licensed under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/
		
THIS DATA IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


