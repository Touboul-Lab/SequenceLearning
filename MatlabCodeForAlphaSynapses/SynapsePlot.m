t=-20:0.01:30


figure;
hold on
tau=3;
plot(t,(t.*exp(-t/tau)/max(t.*exp(-t/tau)).*(t>0)/tau^2))
plot(t,(exp(-t/tau).*(t>0)/tau))

tau=5;
plot(t,(t.*exp(-t/tau).*(t>0)/max(t.*exp(-t/tau))/tau^2))
plot(t,(exp(-t/tau).*(t>0)/tau))

% tau=10;
% plot(t,t.*exp(-t/tau)/max(t.*exp(-t/tau)))
% plot(t,exp(-t/tau))
