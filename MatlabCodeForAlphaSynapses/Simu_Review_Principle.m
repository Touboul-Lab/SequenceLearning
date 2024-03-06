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
epsilon=0.01;
Areward=0.9;
Apostpre=1;
Aprepost=-1;


N_pres=80;

dt=0.1;
ncoeffs=10;
Coeffs=linspace(0.1,6,ncoeffs);

Successes=zeros(ncoeffs,N_pres);
Coeffs=20;
wmax=0.5;
PLOTS=1;
for coco=1:ncoeffs
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
%     wmax=0.2;
    wmin=0;
    
    
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
                Successes(coco,ceil(k/n_Exp))=Successes(coco,ceil(k/n_Exp))-2;
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
end
% figure;
% hold on
% plot(tpost,0*tpost,'o');
% plot(T_pattern+(T_pattern+T_wait)*((1:N_pres)-1),zeros(1,N_pres),'*');
% plot((T_pattern+T_wait)*((1:N_pres)),zeros(1,N_pres),'diamond');
figure;bar(mean(Successes'))