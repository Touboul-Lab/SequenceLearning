clear all
close all
COMMONPLOTS=1;
ALLPLOTS=1;
sigma=0.5;
Veq=-76.72;
Vth=-39.51;
Vr=-41.70;
tau=11.85;
Vsp=-20;

R=118.5;
C=0.098;
tau_Ref=1;


P=10;

taus=20;
epsilon=0.05;

T=20000;
rate=10;

dt=0.1;
probaSpike=rate*(1e-3)*dt;
probaSpikeext=0.1*(1e-3)*dt;


wmax=2;
wmin=0;

t_vect=0:dt:T;
ntimes=length(t_vect);
n_Ref=floor(tau_Ref/dt);

W0=(wmin+wmax)/2;

V=Veq*ones(size(t_vect));

k=1;
I=0*(t_vect<15);

tpost=[];
last_spike=-2*tau_Ref;
Pre_Spikes=zeros(P,ntimes);
Arewards=[0,0.5];
Apreposts=[-1,1];
Apostpres=[-1,1];
Coeffs=[1];
Wext=1;

reps=1;
FiringRates=zeros(reps,8);

for i=1:reps
    FF=zeros(1,8);
    for Coeff=Coeffs
        for Areward=0
            for Aprepost=1
                for Apostpre=-1
                    if Aprepost==-1
                        if Apostpre==-1
                            NAME='LTD';
                            type=1;
                        else
                            NAME='AntiHebbian';
                            type=2;
                        end
                    else
                        if Apostpre==-1
                            NAME='Hebbian';
                            type=3;
                        else
                            NAME='LTP';
                            type=4;
                        end
                    end
                    if Coeff==Coeffs(1)
                        WEIGHTS='Low';
                        W0=Coeff;
                        W=W0*rand(P,ntimes);
                    else
                        WEIGHTS='High';
                        
                        W=Coeffs(1)+(Coeffs(2)-Coeffs(1))*rand(P,ntimes);
                    end
                    if Areward
                        REWARDS='Reward';
                        type=4+type;
                    else
                        REWARDS='NoReward';
                    end
                    
                    
                    V=Veq*ones(size(t_vect));
                    %             if Aprepost==-1
                    %                 if strcmp(NAME,'AntiHebbian')
                    if 1==1
                        k=1;
                        I=0*(t_vect<15);
                        
                        tpost=[];
                        last_spike=-2*tau_Ref;
                        Pre_Spikes=zeros(P,ntimes);
                        
                        while k<ntimes
                            Pre_Spikes(:,k)=rand(1,P)<probaSpike;
                            if (k*dt-last_spike)>tau_Ref
                                V(k+1)=V(k)+dt*(-(V(k)-Veq)+R*tau*(I(k)+Wext*(rand()<probaSpikeext)))/tau+sqrt(dt/tau)*sigma*randn()+sum(R*Pre_Spikes(:,k).*W(:,k));
                            else
                                V(k+1)=Vr;
                            end
                            
                            if (V(k+1)>Vth)
                                V(k)=Vsp;
                                V(k+1)=Vr;
                                last_spike=(k+1)*dt;
                                tpost(end+1)=last_spike;
                                for j=1:P
                                    pre_times=find(Pre_Spikes(j,1:k))*dt;
                                    W(j,k:end)=W(j,k)+epsilon*Aprepost*sum(exp(-(last_spike -pre_times)/taus));
                                    W(j,k:end)=min(wmax,max(wmin,W(j,k:end)));
                                    %             if j==4
                                    % %                 Aprepost*sum(exp(-(last_spike -pre_times)/taus));
                                    %                 (W(j,k)-W(j,k-1))/epsilon
                                    %             end
                                end
                            end
                            
                            for j=1:P
                                if Pre_Spikes(j,k)
                                    W(j,k:end)=W(j,k)+epsilon*Areward;
                                    if tpost<k*dt
                                        W(j,k:end)=W(j,k:end)+epsilon*Apostpre*sum(exp(-(k*dt-tpost)/taus));
                                    end
                                    W(j,k:end)=min(wmax,max(wmin,W(j,k:end)));
                                end
                                if j==4
                                    %             Apostpre*sum(exp(-(k*dt-tpost)/taus))
                                end
                            end
                            
                            k=k+1;
                        end
                        display(strcat(NAME,sprintf(', NumSpikes=%d',(length(tpost)))))
                        FiringRates(i,type)=sum(tpost>T/2);
                        WW=W(:,ceil(end/2):end);
                        if ALLPLOTS
                            bins=linspace(wmin,wmax,30);
                            figure();
                            imagesc(W)
                            caxis([wmin,wmax])
                            %             title(sprintf('Areward=%.1f,Aprepost=%.1f,Apostpre=%.1f,Coeff=%.1f',Areward,Aprepost,Apostpre,Coeff));
                            title(strcat(NAME,' ', REWARDS,' ', WEIGHTS))
                            saveas(gcf,strcat('weights_',NAME,REWARDS,WEIGHTS,'.pdf'))
                            
                            figure()
                            histogram(WW(:),bins)
                            %             title(sprintf('Areward=%.1f,Aprepost=%.1f,Apostpre=%.1f,Coeff=%.1f',Areward,Aprepost,Apostpre,Coeff));
                            title(strcat(NAME,' ', REWARDS,' ', WEIGHTS))
                            saveas(gcf,strcat('histo_',NAME,REWARDS,WEIGHTS,'.pdf'))
                            
                            figure()
                            histogram(WW(:),30)
                            %             title(sprintf('Areward=%.1f,Aprepost=%.1f,Apostpre=%.1f,Coeff=%.1f',Areward,Aprepost,Apostpre,Coeff));
                            title(strcat(NAME,' ', REWARDS,' ', WEIGHTS))
                            saveas(gcf,strcat('histo_',NAME,REWARDS,WEIGHTS,'_freebound.pdf'))
                        end
                        if COMMONPLOTS
                            figure(100)
                            bins=linspace(wmin,wmax,100);
                            hold on
                            histogram(WW(:),bins,'DisplayName',strcat(NAME,' ',REWARDS))
                            
                            
                            figure(101)
                            [a,b]=hist(WW(:),bins);
                            hold on
                            plot(b,a,'DisplayName',strcat(NAME,' ',REWARDS));
                            
                            figure(102);
                            hold on
                            plot(movmean(mean(W),2000))
                            title(strcat(NAME,' ', REWARDS))
                        end
                    end
                end
            end
        end
    end
end
if COMMONPLOTS
    figure(100)
    legend()
    figure(101)
    legend()
    figure(102)
    legend()
end
%%
FiringRatesShaped=FiringRates';
figure;
bar(mean(FiringRatesShaped')*2/(T*1e-3))
hold on
errorbar(mean(FiringRatesShaped')*2/(T*1e-3),std(FiringRatesShaped')*2/(T*1e-3)/sqrt(reps),'.','linewidth',1)
tvals=zeros(8);
for i=1:8
    for j=1:8
        [h,p]=ttest2(FiringRatesShaped(i,:),FiringRatesShaped(j,:));
        tvals(i,j)=p;
    end
end
pscores=[0.05,0.01,0.005,0.001,0.0005];
tt=zeros(8);
for p=pscores
    tt=tt+(tvals<p)
end
figure;
imagesc(tt)