clear all
close all

sigma=0.5;
Veq=-76.72;
Vth=-39.51;
Vr=-41.70;
tau=11.85;

R=118.5;
C=0.098;
tau_Ref=1;


P=10;

taus=20;
epsilon=0.02;
Areward=0.9;
Apostpre=1;
Aprepost=-1;


N_pres=100;

dt=0.1;

% Coeffs=1;
PLOTS=0;
ncoeffs=50;
Coeffs=linspace(0.11,20,ncoeffs);

wmaxmin=0.1;
wmaxmax=0.6;
nwmax=50;
Wmaxes=linspace(wmaxmin,wmaxmax,nwmax);
Heatmap=zeros(ncoeffs,nwmax);
Type1=zeros(ncoeffs,nwmax);
Type2=zeros(ncoeffs,nwmax);
% wmax=0.2;
wmin=0;
parpool(7)
parfor Nwmax=1:nwmax
    wmax=Wmaxes(Nwmax);
    
for coco=1:ncoeffs
    Successes=zeros(ncoeffs,N_pres);
    Coeff=Coeffs(coco);
    
    T_pattern=10*Coeff;
    SpikeTimes=[1 3 5 7]*Coeff;
    Pattern=zeros(P,ceil(T_pattern/dt));
    
    for i=1:length(SpikeTimes)
        Pattern(i,ceil(SpikeTimes(i)/dt))=1;
    end
    
    T_wait=5*T_pattern;
    T=(T_pattern+T_wait)*N_pres;
    
    t_vect=0:dt:T;
    ntimes=length(t_vect);
    n_Ref=floor(tau_Ref/dt);
    
    
    
    W=0.15*ones(P,ntimes);
    V=Veq*ones(size(t_vect));
    
    k=1;
    I=0*(t_vect<15);
    Pre_Spikes=zeros(P,ntimes);
    
    n_Exp=ceil((T_pattern+T_wait)/dt);
    n_Patt=ceil(T_pattern/dt);
    
    
    
    for i=1:N_pres
        k0=(i-1)*ceil((T_pattern+T_wait)/dt)+1;
        k1=(i-1)*ceil((T_pattern+T_wait)/dt)+ceil(T_pattern/dt);
        Pre_Spikes(:,k0:k1)=Pattern;
    end
    
    
    
    
    tpost=[];
    last_spike=-2*tau_Ref;
    % REF=FALSE;
    while k<ntimes
        if (k*dt-last_spike)>tau_Ref
            V(k+1)=V(k)+dt*(-(V(k)-Veq)+R*I(k))/tau+sqrt(dt/tau)*sigma*randn()+sum(R*Pre_Spikes(:,k).*W(:,k));
        else
            V(k+1)=Vr;
        end
        
        if (V(k+1)>Vth)
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
            if mod(last_spike,T_pattern+T_wait)>=max(SpikeTimes)
                Successes(coco,ceil(k/n_Exp))=Successes(coco,ceil(k/n_Exp))+1;
            else
                Successes(coco,ceil(k/n_Exp))=Successes(coco,ceil(k/n_Exp))-3;
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
    if PLOTS
    figure;
    title(sprintf('Coeff=%.2f',(Coeff)));
    % ax1=subplot(3,1,1);
    plot(t_vect,V)
    hold on
    plot(tpost,-40*ones(size(tpost)),'*')
    plot(max(SpikeTimes)+(T_pattern+T_wait)*(0:(N_pres-1)),-40*ones(1,N_pres),'o')
    
    figure;
    % ax2=subplot(3,1,2);
    imagesc(W(:,1:10:end))
    caxis([wmin,wmax]);
    % title(sprintf('Coeff=%.2f',(Coeff)));
    figure;
    % ax3=subplot(3,1,3);
    plot(t_vect,W')
    % title(sprintf('Coeff=%.2f',(Coeff)));
    % Coeff
    % linkaxes([ax1,ax2,ax3],'x');
    end
    Heatmap(coco,Nwmax)=mean(Successes(coco,:)>0);
    Type1(coco,Nwmax)=mean(Successes(coco,:)<0);
    Type2(coco,Nwmax)=mean(Successes(coco,:)==0);
end
% figure;
% hold on
% plot(tpost,0*tpost,'o');
% plot(T_pattern+(T_pattern+T_wait)*((1:N_pres)-1),zeros(1,N_pres),'*');
% plot((T_pattern+T_wait)*((1:N_pres)),zeros(1,N_pres),'diamond');
% figure;bar(mean(Successes'))
end
%%
load longSimsFInalLargeInterval
figure;
pcolor(Wmaxes, 7*Coeffs,Heatmap)
colorbar()
figure;
pcolor(Wmaxes, 7*Coeffs,Type1)
caxis([0 1])
title('Type 1: early spike')
figure;
pcolor(Wmaxes, 7*Coeffs,Type2)
caxis([0 1])
title('Type 2: no spike')
figure;
pcolor(Wmaxes, 7*Coeffs,Heatmap+Type1+Type2)


T=0:0.1:140;
wspike=(Vth-Veq)/R;
NoSpikeAtAll=wspike./(1+exp(-2*T/(7*tau))+exp(-4*T/(7*tau))+exp(-6*T/(7*tau)));
ThreeAndSpike=wspike./(1+exp(-2*T/(7*tau))+exp(-4*T/(7*tau)));
TwoAndSpike=wspike./(1+exp(-2*T/(7*tau)));
OneAndSpike=wspike*ones(size(T));


figure;
pcolor(Wmaxes, 7*Coeffs,Heatmap)
hold on
plot(NoSpikeAtAll,T,'k','linewidth',3);
plot(ThreeAndSpike,T,'r','linewidth',3);
% w=0:0.01:0.6;
% plot(w,-tau*log(w/wspike),'linewidth',3);
% plot(wspi,'linewidth',3);

figure;
pcolor(Wmaxes, 7*Coeffs,Type1)
hold on
plot(0,0);
plot(ThreeAndSpike,T,'r','linewidth',3);
caxis([0 1])
title('Type 1: early spike')
figure;
pcolor(Wmaxes, 7*Coeffs,Type2)
hold on
plot(NoSpikeAtAll,T,'k','linewidth',3);

caxis([0 1])
title('Type 2: no spike')

%%
load longSimsFInalLargeInterval
failureRate=1/(abs(Areward/(Areward+Aprepost))+1);
figure;
histogram(Type2(Type2>0),0.00:0.01:1.1,'Normalization','probability')
hold on
plot([failureRate,failureRate],[0 0.3],'linewidth',2);