cd ~/Dropbox/Projet_Gaetan/Sequence/'NatureCom 2'/AlphaSynapses/Alpha_Synapses/
tau_list=[0.01,0.1,0.5,1,5,10,15,20,25,0];
pattern_list=[5, 10,15,20];
linescolor=['r','g','b','c','m','y','k'];
linestyle={'-','--','-o','-.','-*'}
kt=0;
figure(1);
    hold on;
    Final=zeros(length(tau_list),length(pattern_list));
    Stds=zeros(length(tau_list),length(pattern_list));
    Final_All=zeros(length(tau_list),length(pattern_list),10);
for tau=tau_list
    kt=kt+1;
    if tau==0.01
        cd Tau_0_01/
    elseif tau==0.1
        cd Tau_0_1/
    elseif tau==0.5
        cd Tau_0_5/
    elseif tau==1
        cd Tau_1/
    elseif tau==5
        cd Tau_5/
    elseif tau==10
        cd Tau_10/
    elseif tau==15
        cd Tau_15/
    elseif tau==20
        cd Tau_20/
    elseif tau==25
        cd Tau_25/
    else
        cd Dirac
    end
    
    kn=0;
    for n=pattern_list
        kn=kn+1;
        accuracies=load(sprintf('Accuracies_Alpha%d.txt',n));
        iterations=load(sprintf('IterationTest%d.txt',n));
        figure(1)
        plot(iterations,mean(accuracies),strcat(linescolor(mod(kt,7)+1),linestyle{kn}),'DisplayName',sprintf('tau=%.2f, n=%d',tau,n));
        Final(kt,kn)=mean(accuracies(:,end));
        Stds(kt,kn)=std(accuracies(:,end));
    end
    cd ..
end
ylim([0,1])
legend show

%% 
n=15;
cd Tau_0_5/
accuracies05=load(sprintf('Accuracies_Alpha%d.txt',n));
cd ../Tau_5/
accuracies5=load(sprintf('Accuracies_Alpha%d.txt',n));
%%
figure;
hold on
kt=0;
for tau=tau_list
    kt=kt+1
    bar((1:4)+0.09*(kt-1),Final(kt,:),0.09,'DisplayName',sprintf('tau=%.2f',tau))
    errorbar((1:4)+0.09*(kt-1),Final(kt,:),Stds(kt,:)/sqrt(40),'o')
end
bar((1:4)+0.09*(kt-1),Final(kt,:),0.09,'k')
legend show
xlim([0.9 5])
%%
nt=length(tau_list);
nn=length(pattern_list);
pvals=zeros(nn,nt,nt);
hvals=zeros(nn,nt,nt);
n=0;
cd ~/Dropbox/Projet_Gaetan/Sequence/'NatureCom 2'/AlphaSynapses/Alpha_Synapses/

for pt=pattern_list
    n=n+1;
    n1=0;
    for tau1=tau_list
        n1=n1+1;
        if tau1==0.01
            cd Tau_0_01/
        elseif tau1==0.1
            cd Tau_0_1/
        elseif tau1==0.5
            cd Tau_0_5/
        elseif tau1==1
            cd Tau_1/
        elseif tau1==5
            cd Tau_5/
        elseif tau1==10
            cd Tau_10/
        elseif tau1==15
            cd Tau_15/
        elseif tau1==20
            cd Tau_20/
        elseif tau1==25
            cd Tau_25/
        else
            cd Dirac/
        end
        accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pt));
        cd ..
        n2=0;
        for tau2=tau_list
        n2=n2+1;
        if tau2==0.01
            cd Tau_0_01/
        elseif tau2==0.1
            cd Tau_0_1/
        elseif tau2==0.5
            cd Tau_0_5/
        elseif tau2==1
            cd Tau_1/
        elseif tau2==5
            cd Tau_5/
        elseif tau2==10
            cd Tau_10/
        elseif tau2==15
            cd Tau_15/
        elseif tau2==20
            cd Tau_20/
        elseif tau2==25
            cd Tau_25/
        else
            cd Dirac2/
        end
        accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pt));
            [h,p]=ttest2(accuracies1(:,end),accuracies2(:,end));
            pvals(n,n1,n2)=p;
            hvals(n,n1,n2)=h;
            cd ..
        end
    end
end

NumStars=(pvals<0.05)+(pvals<0.01)+(pvals<0.005)+(pvals<0.001)+(pvals<0.0005);
% NumStars=(pvals<0.05)+(pvals<0.005)+(pvals<0.0005);
% NumStars=(pvals<0.05)+(pvals<0.01)+(pvals<0.001)+(pvals<0.0001);
% close all
% figure;imagesc(squeeze(pvals(1,:,:)))
% figure;imagesc(squeeze(hvals(1,:,:)))
% 
% figure;imagesc(squeeze(pvals(2,:,:)))
% figure;imagesc(squeeze(hvals(2,:,:)))
% 
% figure;imagesc(squeeze(pvals(3,:,:)))
% figure;imagesc(squeeze(hvals(3,:,:)))
% 
figure;imagesc(squeeze(pvals(4,:,:)))
% figure;imagesc(squeeze(hvals(4,:,:)))

figure;imagesc(squeeze(NumStars(1,:,:)))
title(sprintf('Number of Patterns: %d',pattern_list(1)))
figure;imagesc(squeeze(NumStars(2,:,:)))
title(sprintf('Number of Patterns: %d',pattern_list(2)))
figure;imagesc(squeeze(NumStars(3,:,:)))
title(sprintf('Number of Patterns: %d',pattern_list(3)))
figure;imagesc(squeeze(NumStars(4,:,:)))
title(sprintf('Number of Patterns: %d',pattern_list(4)))
%%
cd ~/Dropbox/Projet_Gaetan/Sequence/'NatureCom 2'/AlphaSynapses/Alpha_Synapses/
tau_list=[0.01,0.1,0.5,1,5,10,15,20,25,0];
pattern_list=[5, 10,15,20];
linescolor=['r','g','b','c','m','y','k'];
linestyle={'-','--','-o','-.','-*'}
kt=0;
figure(1);
    hold on;
    Final_Exp=zeros(length(tau_list),length(pattern_list));
    Stds_Exp=zeros(length(tau_list),length(pattern_list));
    Final_All=zeros(length(tau_list),length(pattern_list),10);
for tau=tau_list
    kt=kt+1;
    if tau==0.01
        cd Exponential_Tau_0_01/
    elseif tau==0.1
        cd Exponential_Tau_0_1/
    elseif tau==0.5
        cd Exponential_Tau_0_5/
    elseif tau==1
        cd Exponential_Tau_1/
    elseif tau==5
        cd Exponential_Tau_5/
    elseif tau==10
        cd Exponential_Tau_10/
    elseif tau==15
        cd Exponential_Tau_15/
    elseif tau==20
        cd Exponential_Tau_20/
    elseif tau==25
        cd Exponential_Tau_25/
    else
        cd Dirac2
    end
    
    kn=0;
    for n=pattern_list
        kn=kn+1;
        accuracies=load(sprintf('Accuracies_Alpha%d.txt',n));
        iterations=load(sprintf('IterationTest%d.txt',n));
        figure(1)
        plot(iterations,mean(accuracies),strcat(linescolor(mod(kt,7)+1),linestyle{kn}),'DisplayName',sprintf('tau=%.2f, n=%d',tau,n));
        Final_Exp(kt,kn)=mean(accuracies(:,end));
        Stds_Exp(kt,kn)=std(accuracies(:,end));
    end
    cd ..
end
ylim([0,1])
legend show

%%
figure;
hold on
kt=0;
tau_list=[0.01,0.1,1,5,10,15,20,25];
for tau=tau_list
    kt=kt+1
    bar((1:4)+0.09*(kt-1),Final_Exp(kt,:),0.09,'DisplayName',sprintf('tau=%.2f',tau))
    errorbar((1:4)+0.09*(kt-1),Final_Exp(kt,:),Stds_Exp(kt,:)/sqrt(40),'o')
end
bar((1:4)+0.09*(kt-1),Final(kt,:),0.09,'k')
legend show
xlim([0.9 5])

figure;
hold on
kt=0;
for tau=tau_list
    kt=kt+1
    bar((1:4)+0.09*(kt-1),Final(kt,:),0.04,'DisplayName',sprintf('tau=%.2f',tau))
    bar((1:4)+0.09*(kt-0.5),Final_Exp(kt,:),0.04,'DisplayName',sprintf('tau=%.2f',tau))
end

%%
N=100;
Colors=jet(N);
Intervalues=linspace(min(min(Final(:)),min(Final(:)))-0.1,max(max(Final(:)),max(Final(:)))+0.1,N);
Intervalues=linspace(0,1,N);
tau_list=[0.01,0.1,0.5,1,5,10,15,20];
figure()
hold on
Pvals=zeros(4,7);
cd ~/Dropbox/Projet_Gaetan/Sequence/'NatureCom 2'/AlphaSynapses/Alpha_Synapses/Exponential_Tau_0_01

for i=1:(length(pattern_list))
    for j=1:(length(tau_list))
%         if j>2
%             jj=j+1;
%         else
%             jj=j;
%         end
        TriangleTopX=[i-1 i-1 i];
        TriangleTopY=[j-1 j j-1];
        TriangleBottomX=[i i-1 i];
        TriangleBottomY=[j-1 j j];
        colR=interp1(Intervalues,Colors(:,1),Final(j,i));
        colG=interp1(Intervalues,Colors(:,2),Final(j,i));
        colB=interp1(Intervalues,Colors(:,3),Final(j,i));
        COL=[colR colG colB];
        fill(TriangleTopX,TriangleTopY,COL);
        
        colR=interp1(Intervalues,Colors(:,1),Final_Exp(j,i));
        colG=interp1(Intervalues,Colors(:,2),Final_Exp(j,i));
        colB=interp1(Intervalues,Colors(:,3),Final_Exp(j,i));
        COL=[colR colG colB];
        
        fill(TriangleBottomX,TriangleBottomY,COL)
        if j==1
            cd ../Exponential_Tau_0_01/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_0_01/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==2
            cd ../Exponential_Tau_0_1/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_0_1/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==3
            cd ../Exponential_Tau_0_5/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_0_5/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==4
            cd ../Exponential_Tau_1/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_1/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==5
            cd ../Exponential_Tau_5/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_5/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==6
            cd ../Exponential_Tau_10/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_10/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==7
            cd ../Exponential_Tau_15/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_15/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        elseif j==8
            cd ../Exponential_Tau_20/
            accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
            cd ../Tau_20/
            accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%         elseif j==3
%             cd ../Exponential_Tau_1/
%             accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%             cd ../Tau_1/
%             accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%         elseif j==4
%             cd ../Exponential_Tau_5/
%             accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%             cd ../Tau_5/
%             accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%         elseif j==5
%             cd ../Exponential_Tau_10/
%             accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%             cd ../Tau_10/
%             accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%         elseif j==6
%             cd ../Exponential_Tau_15/
%             accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%             cd ../Tau_15/
%             accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%         elseif j==7
%             cd ../Exponential_Tau_20/
%             accuracies1=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
%             cd ../Tau_20/
%             accuracies2=load(sprintf('Accuracies_Alpha%d.txt',pattern_list(i)));
        end
        [h,p]=ttest2(accuracies1(:,end),accuracies2(:,end));
        if abs(Final(j,i)-mean(accuracies2(:,end)))>0
            fprintf('Wow wow, problème\n');
        end
        Pvals(i,j)=p;
        if p<0.05
            scatter(i-1/2,j-1/2,40,'*','k','linewidth',1.5)
            fprintf('Significant!,p=%f',p)
        end
        if p<0.005
            scatter(i-1/2-0.1,j-1/2,'*','k')
            fprintf('Very Significant!')
        end
        if p<0.0005
            scatter(i-1/2+0.1,j-1/2,'*','k')
            fprintf('Very Very Significant!')
        end
    end
end
colormap(jet)
colorbar()
%%
find(Final(:,1)==max(Final(:,1)))
find(Final(:,2)==max(Final(:,2)))
find(Final(:,3)==max(Final(:,3)))
find(Final(:,4)==max(Final(:,4)))

find(Final_Exp(:,1)==max(Final_Exp(:,1)))
find(Final_Exp(:,2)==max(Final_Exp(:,2)))
find(Final_Exp(:,3)==max(Final_Exp(:,3)))
find(Final_Exp(:,4)==max(Final_Exp(:,4)))