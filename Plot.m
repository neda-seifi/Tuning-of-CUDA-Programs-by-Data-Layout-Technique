format long;
clc;
disp('Plot:');
%------------------------------------------------------- Reading from File
fileID = fopen('output.txt');
M = fscanf(fileID, '%d %f %f %f',[4 inf]);
fclose(fileID);
M=M';

n=M(1:400,1:1); 
time1=M(1:400,2:2);   
time2=M(1:400,3:3);   
time3=M(:,4:4);   


figure
%time1=movmean(time1,5);
%plot(n,time1),xlabel('Data Size (n)'),ylabel('Execution Time (msec)'),title('Compare');

plot(n,time1,n,time2,'linewidth',2),xlabel('Data Size (n)'),ylabel('Execution Time (msec)'),title('Comparision of serial and CUDA version');
legend('Serial version','CUDA version');

% speedup=time1;
% for i=1:length(time1)
%       speedup(i)=time1(i)/time2(i);
% end
% plot(n,speedup,'linewidth',2),xlabel('Data Size (n)'),ylabel('Speedup'),title('Speedup for Serial version/CUDA version');

%speedup=time1/time2;
%plot(n,speedup,'linewidth',2),xlabel('Data Size (n)'),ylabel('Speedup'),title('Speedup for Serial version/CUDA version');

%plot(n,time2,n,time3,'linewidth',2),xlabel('Data Size (n)'),ylabel('Execution Time (msec)'),title('Comparision of CUDA version and Tuned CUDA version');
%legend('CUDA version','Tuned CUDA version');

% speedup=time2;
% for i=1:length(time2)
%      speedup(i)=time2(i)/time3(i);
% end
% plot(n,speedup,'linewidth',2),xlabel('Data Size (n)'),ylabel('Speedup'),title('Speedup for CUDA version/Tuned CUDA version');

