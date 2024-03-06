%%%%% Neuron Parameters %%%%%%
C=50;
k_input=1.;
v_rest=-80.;
v_t=-20.;
c=-55.;
v_peak=40.;
a=0.01;
b=-20.;
d=150.;
noise=0;
R=100;

%%%%% Synapse Parameter %%%%%%
tau_s=20;
Apostpre=1;

%%%%%% Simulation Parameters %%%%%

T=300;
dt=0.01;
t_vect=0:dt:T;
n_times=length(t_vect);
ve=zeros(1,n_times);
we=zeros(1,n_times);
ve(1)=v_rest;

va=zeros(1,n_times);
wa=zeros(1,n_times);
va(1)=v_rest;

vd=zeros(1,n_times);
wd=zeros(1,n_times);
vd(1)=v_rest;

% I=R*C*I;

%%%%%% Input %%%%%%
P=10;
taus=10;
W=zeros(P,1);
tspike_list=[100 120 125 126 200];
% tspike_list=[100 ];
I0=6000;
SynapseAlpha=@(t)I0*t.*exp(-t/taus).*(t>0)/taus^2;
SynapseExp=@(t)I0*exp(-t/taus).*(t>0)/taus;
SynapseDirac=@(t)I0*(t==0)/dt;
IA=zeros(size(t_vect));
IE=zeros(size(t_vect));
ID=zeros(size(t_vect));
for t_spike=tspike_list
    t=t_vect-t_spike;
    IA=IA+SynapseAlpha(t);
    IE=IE+SynapseExp(t);
    ID=ID+SynapseDirac(t);
end
% figure;
% hold on
% tau=3;
% plot(t,(t.*exp(-t/tau).*(t>0)/tau^2))
% plot(t,(exp(-t/tau).*(t>0)/tau))
% 
% tau=5;
% plot(t,(t.*exp(-t/tau).*(t>0)/tau^2))
% plot(t,(exp(-t/tau).*(t>0)/tau))

refrac=0;
n_refrac=0;


%%%%%% Time Stepping %%%%%
for k=1:(n_times-1)
    ve(k+1)=ve(k)+dt*(f(ve(k),IE(k),k_input,v_rest,v_t)-we(k))/C + noise*sqrt(dt/C)*randn();
    we(k+1)=we(k)+dt* a * (b* (ve(k)- v_rest) - we(k));
    if (ve(k+1)>v_peak)
        we(k+1)=we(k+1)+d;
        ve(k)=v_peak;
        ve(k+1)=c;
    end
        va(k+1)=va(k)+dt*(f(va(k),IA(k),k_input,v_rest,v_t)-wa(k))/C + noise*sqrt(dt/C)*randn();
    wa(k+1)=wa(k)+dt* a * (b* (va(k)- v_rest) - wa(k));
    if (va(k+1)>v_peak)
        wa(k+1)=wa(k+1)+d;
        va(k)=v_peak;
        va(k+1)=c;
    end
    vd(k+1)=vd(k)+dt*(f(vd(k),ID(k),k_input,v_rest,v_t)-wd(k))/C + noise*sqrt(dt/C)*randn();
    wd(k+1)=wd(k)+dt* a * (b* (vd(k)- v_rest) - wd(k));
    if (vd(k+1)>v_peak)
        wd(k+1)=wd(k+1)+d;
        vd(k)=v_peak;
        vd(k+1)=c;
    end

end
% figure;
% plot(v,w)
figure;
plot(t_vect,va)
hold on
plot(t_vect,ve)
plot(t_vect,vd)

ylim([-150,50])

figure;
plot(t_vect,IA)
hold on
plot(t_vect,IE)
% plot(t_vect,ID)
ylim([-1,14000])


function dv=f(v, I, k_input,v_rest,v_t)
    dv = k_input*(v_rest - v) * (v_t-v) + I;
end
