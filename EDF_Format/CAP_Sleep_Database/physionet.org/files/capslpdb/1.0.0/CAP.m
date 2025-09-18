function [CAPtime, rate]=CAP(time_tot, duration, hyp)

% This function applies Terzano's rules [Terzano et al., 2001] to count CAP sequences
% and computes for each night the CAPtime and the CAPrate=CAPtime/NREMtime
% Developed by Martin O. Mendez, PhD and Sara Mariani, MSc, under the supervision of professor Anna M. Bianchi and Professor 
% Sergio cerutti, at the Department of Bioengineering of Politecnico di Milano, Italy

% inputs:
% - time_tot is the vector of the starting times of phases A in seconds
% - duration is the vector of the durations of phases A in seconds
% - hyp is a two-column matrix containing the hypnogram and the reference epochs

startB =[];
startA = [];
durationB = [];
upto60 = [];
CAPs=[];
CAPf=[];

startB = time_tot+duration; %B phase start
startA = time_tot;           %A phase start
durationB = startA(2:end)-startB(1:end-1);  %array containing the durations of B phases
upto60 = find(durationB>=60);               %non CAP

count=0;
countCAP=0;
for k = 1:length(durationB),    %for each B phase
      
    if durationB(k)<60,
        count = count + 1;      
    else
        if count>=2,
            countCAP=countCAP+1;      %number of CAP sequences
            CAPf(countCAP)=k;         %correspondance between the number or CAP sequence found and the phase A in which it ended
            CAPs(countCAP)=k-count;   %correspondance between the number or CAP sequence found and the phase A in which it started
        end
        count=0;
    end
    if k==length(durationB) & count>=2,
            countCAP=countCAP+1;
            CAPf(countCAP)=k+1;
            CAPs(countCAP)=k-count+1;
    end
end

NREMtime=numel(find(hyp(:,1)~=5 & hyp(:,1)~=0))*30;      %%sleep NREM time
CAPtime=sum(startA(CAPf)-startA(CAPs));
rate=CAPtime/NREMtime;
