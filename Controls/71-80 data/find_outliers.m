function [lst_of_outliers]=find_outliers(limit,filename)
[num,txt,raw]=xlsread(filename);
lst_of_outliers=[];
for i=1:(length(num(:,1)))
    if  num(i,14)>limit %|| num(i,15)>limit 
        lst_of_outliers=[lst_of_outliers,raw(i+1,2)];
    end
end
end
