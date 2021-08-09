%% Non-dimensionalized Neuronal Dynamical System %%
clc
clear
close all

%% System Constants and Parameters
nL = 2048;      % # Neurons in LGN

%% Learning Data
% TBD

%% Retina Structure Parameters
%{
Ret = {};       % Retina Data Structure
% centroid_RGC = mean(Ret.nx); dist_center_to_all = pdist2(Ret.nx, centroid_RGC); gaussian_val = 6*exp(-(dist_center_to_all)/10);

Ret.v_reset = 0 + 0.1*randn(nR,1).^2;     %Noise on activity field

% Spatial Parameters
sqR = 28; Ret.nx = sqR*rand(nR,2); Ret.D = squareform(pdist(Ret.nx)); %Distance matrix

% Adjacency kernel
Ret.ri = 3; Ret.ro = 6.0; Ret.lam = 10; %Inner/outer radius & decay char. length of kernel
Ret.ai = 30; Ret.ao = 10;               %Excitation & inhibition amplitude
Ret.S = Ret.ai*(Ret.D < Ret.ri)- Ret.ao*(Ret.D > Ret.ro).*exp(-Ret.D / Ret.lam); 
Ret.S = Ret.S - diag(diag(Ret.S));      %Adjacency matrix between Neurons in Retina

% Dynamics Parameters
Ret.tau_v = 1; 
Ret.tau_th = 30; Ret.th_plus = 9; Ret.v_th = 1;

% Initializations of I.C.:
Ret.v  = 0*ones(nR,1); Ret.u  = 0*ones(nR,1); Ret.th  = ones(nR,1);
Ret.H = sparse(zeros(nR,1)); %"spikeMat"
Ret.htmp = zeros(nR,1); %Heatmap # of times each neuron spikes
%}

%% LGN (L2) Structure Parameters
parameter_property_matrix = zeros(8000,12)


paramater_property_vec = zeros(1,12)
LGN = {};       % LGN Data Structure
LGN.v_reset = 0 + 0.1*randn(nL,1).^2;     %Noise on activity field
disp('i')
disp(i)
% Spatial Parameters
%% sql is the dimensions of the aquare the neural net exists in.
%% LGN.nx creates the positions of all the neurons
%% LGN D gives the distance between each neuron and all the other neurons in the network.
sqL = 20; LGN.nx = sqL*rand(nL,2); LGN.D = squareform(pdist(LGN.nx)); %Distance matrix
disp('dimensions of LGN.x')
disp(ndims(LGN.nx))
disp(LGN.nx)
% Adjacency kernel
LGN.ri = 1.0; LGN.ro = 1.5; LGN.lam = 10; %Inner/outer radius & decay char. length of kernel
LGN.ai = 10; LGN.ao = 1.5;               %Excitation & inhibition amplitude
LGN.S = LGN.ai*(LGN.D < LGN.ri)- LGN.ao*(LGN.D > 2*LGN.ro).*exp(-LGN.D / LGN.lam); 
%%diag(diag(LGN.S)) is there to subrtract away the diagonal of the S matrix, since it should be 0
%% (A neuron cannot interact with itself.)
LGN.S = LGN.S - diag(diag(LGN.S));      %Adjacency matrix between Neurons in Retina

% Dynamics Parameters
LGN.tau_v = 1; 
LGN.tau_th = 2; LGN.th_plus = 2; LGN.v_th = 5;

% Initializations of I.C.:
%%initializes the activation of the LGN neurons
%%initializes the input energy of the neurons (the amount of activation they feel from other neurons)
%%initializes the firing threshold of the neurons
LGN.v  = 0*ones(nL,1); LGN.fb  = 0*ones(nL,1); LGN.th  = ones(nL,1);
%% creates a vector to track which neurons are firing or not. 
LGN.H = sparse(zeros(nL,1)); %"spikeMat"
LGN.htmp = zeros(nL,1); %Heatmap # of times each neuron spikes

%% Nondimensional Parameters
% Reference variables
Vt = LGN.ao; Tt = LGN.tau_v; Lt = sqL;




a = .1 + 2*rand() 
b = .3 + 2.2*rand()
c = abs(2-sqrt(-4*(rand()-1.3))) + abs(randn()/4)
d = rand()*5 
e = .5 + 1.5*rand()
f = abs(1 + randn()*.4)
g = f * (1+rand())
h = rand()*1 + randn()*.5



% Non-dimensional groups
%%Mixed/
piTV_plus = a*LGN.th_plus * Tt / Vt
%%Temporal
piT_th =  b*Tt / LGN.tau_th 

%%Activation
piV_th = c*LGN.v_th / Vt
piV_reset = d*LGN.v_reset / Vt;
piV_ai = e*LGN.ai / Vt
%%Topolgical
piL_ri = f*LGN.ri / Lt
piL_ro = g*LGN.ro / Lt
piL_lam = h*LGN.lam / Lt

% Non-dimensional Spatial Adjacency matrix
LGN.nxs = LGN.nx / Lt;
LGN.Ds = squareform(pdist( LGN.nxs));      %Non-dimensional distance matrix
LGN.Ss = piV_ai *(LGN.Ds < piL_ri)- (LGN.Ds > piL_ro).*exp(-LGN.Ds / piL_lam); 
LGN.Ss = LGN.Ss - diag(diag(LGN.Ss));      %Adjacency matrix between Neurons in Retina

% Reference Kernel vs. Non-dimensional Kernel
% figure
% 
% subplot(1,2,1), plot([0:0.01:sqL], LGN.ai*([0:0.01:sqL] < LGN.ri)- LGN.ao*([0:0.01:sqL] > LGN.ro).*exp(-[0:0.01:sqL] / LGN.lam)),xlim([0,sqL]), title('Ref. Kernel')
% subplot(1,2,2), plot([0:0.001:1],piV_ai *([0:0.001:1] < piL_ri)- ([0:0.001:1] > piL_ro).*exp(-[0:0.001:1] / piL_lam)), title('Non-dim. Kernel')

LGN.vs = 0*ones(nL,1)/Vt; LGN.fbs = 0*ones(nL,1)/Vt; LGN.ths = ones(nL,1)/Vt;
LGN.Hs = sparse(zeros(nL,1)); LGN.htmps = zeros(nL,1);

%% Time loop settings & Initialization
% disp('LGN-, LGN-, V1-densities (#neurons/area):'), disp(num2str([nL/sqL^2; nL/sqL^2; nV/sqV^2]))
%% end is the how long the SNN runs for
%% dt is the time step that's used (i think)
%% tint is the time to the end as an interval
%% and ts is the start time of the network
Tend = 3e2; dt = 0.1; tint = 0:Tend/1; ts = 0;
fnoise = 10*randn(nL,1*length(tint)); % Pre-generate noise
% fnoise = X_input;
% fnoiseI = interp1(tint',fnoise',[0:dt:Tend]','linear')';

%%Creates two matricies each for activation, ?????????????, and firing threshold. 
%% each matrix pair represents either the dimensional version or the dimensionless version.
Xv = zeros(nL,tint(end)); Xvs = zeros(nL,tint(end)); %Snapshot matrix v
Xu = zeros(nL,tint(end)); Xus = zeros(nL,tint(end)); %Snapshot matrix u
Xth = zeros(nL,tint(end)); Xths = zeros(nL,tint(end)); %Snapshot matrix th

fireL = []; fireRs = [];
firedMat = cell(1,length(tint)); firedMats = cell(1,length(tint)); 
tic
%Zack Code Start:
LGN.old_Hs = zeros(nL,1)
avg_wave_size = 0
avg_wave_size_volatility = 0
wave_distributivity = 0
wave_distributivity_vec = zeros(nL,1)
avg_wave_neuron_flux = 0

old_wave_id = zeros(nL,1)
%:Zack Code End


%% creates a for loop that takes time step of dt. 
for tt = 0:dt:Tend-dt
%     fprintf('tt: %d\n' , tt)
%     fprintf('i: %d7n' , i)
    LGN.eta = fnoise(:,round(tt+1)); 
%     LGN.eta = X_input(:,round(tt/5)+1);
    LGN.etas = fnoise(:,round(tt+1)) / Vt; 
    
    % Solve Wave Dynamical System in Retina
    %% Finds which neurons can feel the other firing neurons
    %% and how they're effected
    LGN.fb = LGN.S*LGN.H;
    %% Solve the differential equation for voltage
    LGN.v  = RK4(@(v) 1/LGN.tau_v*(-v + LGN.fb + LGN.eta),    dt,LGN.v);
    %% Solve the differential equation for the firing threshold.
    LGN.th = RK4(@(th)(1/LGN.tau_th*(LGN.v_th-th).*(1-LGN.H) + LGN.th_plus*LGN.H),    dt,LGN.th);
    
    % Discontinuous Update rule
    %%No fire L is a vector of all the indicis 
    %% of the currently firing neurons.
    fireL = find(LGN.v >= LGN.th); tmpR = LGN.v;
%{
    % Determine winner(s) of the firing neurons
    [winf, maxIndf] = maxk(LGN.v(fireL),length(fireL));
    LGN.y = sparse(zeros(nL,1));
    LGN.y(fireL(maxIndf)) = LGN.v(fireL(maxIndf))/max([winf; eps]);
%}    
    %%Sets the value of the firing neurons to the reset voltage
    LGN.v(fireL) = LGN.v_reset(fireL);
    %% Resets LGN.H
    LGN.H = sparse(zeros(nL,1));
    %% ): no clue here bud
    LGN.H(fireL,1) = ones(length(fireL),1);

    % NON-DIMENSIONALIZED Wave DS
    LGN.fbs = LGN.Ss*LGN.Hs;
    LGN.vs  = RK4(@(vs) (-vs + LGN.fbs + LGN.etas),    dt, LGN.vs);
    LGN.ths = RK4(@(ths)(piT_th *(piV_th -ths).*(1-LGN.Hs) + piTV_plus *LGN.Hs),    dt/Tt, LGN.ths);
    
    % NON-DIMENSIONALIZED Update rule
    fireRs = find(LGN.vs >= LGN.ths); tmpRs = LGN.vs;
    
%{
    % Determine winner(s) of the firing neurons
    [winf, maxIndf] = maxk(LGN.vs(fireRs),length(fireRs));
    LGN.ys = sparse(zeros(nL,1));
    LGN.ys(fireRs(maxIndf)) = LGN.vs(fireRs(maxIndf))/max([winf; eps]);
%}    
    LGN.vs(fireRs) = piV_reset(fireRs);
    LGN.Hs = zeros(nL,1);
    LGN.Hs(fireRs,1) = ones(length(fireRs),1);
%     disp('LGN.Hs')
%     disp(LGN.Hs)
    if tt == 40
        
    end
    %_% Zack Code Begin: 
%     disp('dimensions of LGN.nx')
%     disp(ndims(LGN.nx))
%     disp(size(LGN.nx))
%     disp('dimensions of LGN.nx(fireRS)')
%     disp(size(LGN.nxs(fireRs,:)))
%     disp(size(LGN.nxs(fireRs,:)))
%     disp('LGN.nxs(fireRs,:)')
%     disp(LGN.nxs(fireRs,:))
    
%     if size(LGN.nxs(fireRs,:)) == size([1,2])
%         position = LGN.nxs(fireRs,:)
%     else
%         position = mean(LGN.nxs(fireRs,:));
%     end
    
%     disp('dimensions of position')
%     disp(ndims(position))
%     disp(size(position))
%     disp('position')
%     disp(position)
%     disp('size(position)')
%     disp(size(position))
%     disp('position')
%     disp(position)
    disp('tt')
    disp(tt)
% if tt == 0
%         position_list = zeros(1,2);
%     else
%         position_list = [position_list; position];
%     end 
%     
    if mod(tt,1) == 0
        Xv(:,ts+1) = tmpR; Xu(:,ts+1) = LGN.fb; Xth(:,ts+1) = LGN.th;
        Xvs(:,ts+1) = tmpRs; Xus(:,ts+1) = LGN.fbs; Xths(:,ts+1) = LGN.ths;
        firedMat{ts+1} = fireL; firedMats{ts+1} = fireRs; 
        ts = ts+1;
        LGN.htmp(fireL) = LGN.htmp(fireL)+1; LGN.htmps(fireRs) = LGN.htmps(fireRs)+1;
    end
    
    
    
    %start Zack Code
    if true
        %get the total wave_size
        avg_wave_size = avg_wave_size + sum(LGN.Hs)/nL;
        if tt ~= 0
            if sum(LGN.old_Hs) ~= 0 
            %get the total wave volatility
            avg_wave_size_volatility = avg_wave_size_volatility + (abs(sum(LGN.Hs) - sum(LGN.old_Hs)))/(sum(LGN.old_Hs));
            %making sure LGN.Hs is not 0.
            end 
            if ~sum(LGN.Hs) == 0
                %get the total neuron flux
                avg_wave_neuron_flux = avg_wave_neuron_flux + abs(sum(abs(LGN.Hs - LGN.old_Hs)) - abs(sum(LGN.Hs)-sum(LGN.old_Hs)) ) / sum(LGN.Hs) ;
            end
        end
        %get a distribution of where neurons fire
        wave_distributivity_vec = wave_distributivity_vec + LGN.Hs;

%         if wave_continuity_implementation == true
%             wave_ID = wave_continuity(LGN.Hs, LGN.Ss, ID_old, ID_taken)
% 
%             ID_list = unique(wave_ID)
%             avg_wave_count = avg_wave_count + numel(unique(wave_ID))
%             for jj = 1:numel(unique(wave_ID))
%                 ID = ID_list(jj) 
                

       
    end 
    %New Wave Continuity
%     if tt == 200
%         try
%         ID_old = zeros(1,nL);
%         ID_taken = zeros(1,nL)
%         [wave_ID_list] = wave_continuity_2(LGN.Hs,LGN.Ss, ID_old,ID_taken)
%         disp('wave_ID_list')
%         disp(wave_ID_list)
%         disp('list of IDs')
%         disp(unique(wave_ID_list))
%         disp('size of each wave:')
%         disp(histc(wave_ID_list , unique(wave_ID_list)))
%         input('halt')
%         catch EE
%             disp(EE)
%             input('Halt, something failed lets go see what it was')
%             
%         end 
%     end
    %WAVE CONTINUITY:
    
%     if tt == 200
%     wave_id = zeros(nL,1);
%     id_counter = 0;
%     for neuron = 1:nL;
%         disp(sprintf('neuron: %d', neuron))
%         if LGN.Hs(neuron) == 1;
% 
%             if wave_id(neuron) == 0;
%                 id_counter = id_counter + 1;
%                 recursion_depth = 0;
%                 disp(sprintf('into the function: ID = %d' , id_counter))
%                 wave_id = wave_continuity(neuron, wave_id, old_wave_id,LGN.Ss, LGN.Hs,id_counter,recursion_depth);
%             else
%                 disp('neuron already had an id')
%             end
%         else
%             disp('neuron not firing')
%         end
%     end
%     disp('number of neurons in wave 1:')
%     disp(sum(wave_id(:)==1))
%     disp('number of neurons in wave 2:')
%     disp(sum(wave_id(:)==2))
%     disp('number of neurons in wave 3:')
%     disp(sum(wave_id(:)==3))
%     
%         gg = input('Halt');
%         old_wave_id = wave_id;
LGN.old_Hs = LGN.Hs;
    end
    %set old_Hs
    
    %set old_wave_id
    %:zack code end

toc
disp('Spatio-temp wave computed') 
%Zack Code Start:
%average things
avg_wave_size = avg_wave_size / (Tend/dt -1);
avg_wave_volatility = avg_wave_size_volatility / (Tend / dt - 2); 
avg_wave_neuron_flux = avg_wave_neuron_flux / (Tend / dt - 2);
%find the variance
wave_distributivity = var(wave_distributivity_vec);
%display results

%% Save These Variables
%What fraction of the total neurons are currently firing
avg_wave_size

%How much bigger or smaller (by number of neurons) is the new wave, 
%then it was 1 timestep Prior?
avg_wave_volatility

% What fraction of newly firing neurons (adjusting for raw change in size)
% weren't firing previously? An approximation for the velocity of the wave. 
avg_wave_neuron_flux

% How unbiased was the wave in its position?
% High wave distributivity means the wave favored firing on the same
% neurons regularly. Low distributivity means its position was more evenly
% spread out.
wave_distributivity

% 
% paramater_property_vec = [a , b , c , d, e ,f, g, h, avg_wave_size, avg_wave_volatility, avg_wave_neuron_flux, wave_distributivity]
% paramater_property_mat(i,:) = paramater_property_vec







%:Zack Code ENd% 
%figure(2),scatter(LGN.nx(:,1),LGN.nx(:,2),[],LGN.htmp(:,end),'filled'), colorbar, title('Heat map - Retina Wave'), axis ij image 
% figure(3),scatter(LGN.nxs(:,1),LGN.nxs(:,2),[],LGN.htmps(:,end),'filled'), colorbar, title('Heat map - Non-dim. Retina Wave'), axis ij image 
% 
% figure(4),subplot(1,3,1),plot(tint(1:end-1),Xv(1:40:end,:),'-o');                          title('LGN.v')
%         subplot(1,3,2),plot(tint(1:end-1),Xu(1:40:end,:),'-o');                            title('LGN.fb')
%         subplot(1,3,3),plot(tint(1:end-1),Xth(1:40:end,:),'-o');                           title('LGN.th')
%         
% figure(5),subplot(1,3,1),plot(tint(1:end-1),Xvs(1:40:end,:),'-o');                          title('Non-dim LGN.vs')
%         subplot(1,3,2),plot(tint(1:end-1),Xus(1:40:end,:),'-o');                            title('Non-dim LGN.fbs')
%         subplot(1,3,3),plot(tint(1:end-1),Xths(1:40:end,:),'-o');                           title('Non-dim LGN.ths')
% 
%% Visualization of Wave
halt = true
for ii = 1:4:length(tint)-1
%     if ii > 200
%         if halt == true
%            input('halt')
%            halt = false
%         end 
%     end
%     f = figure(7); 
%     f.Position(1:4) = [0 0 1600 800];
% % subplot(1,2,1), title(['Non-dim. Wave t = ' num2str(ii)]), axis ij image, hold on,
%     scatter(LGN.nxs(:,1),LGN.nxs(:,2),'k','filled'); scatter(position_list(4*ii,1),position_list(4*ii,2),'r','filled');
%     subplot(1,2,2), title(['Non-dim. Wave t = ' num2str(ii)]), axis ij image, hold on,
%     scatter(LGN.nxs(:,1),LGN.nxs(:,2),'k','filled'); scatter(LGN.nxs(firedMats{ii},1),LGN.nxs(firedMats{ii},2),'r','filled');
%                                              
% % %        
% % %           gifname2='wave_VS_nondim_wave.gif'; frame = getframe(gcf);  im = frame2im(frame);  [imind,cm] = rgb2ind(im,256);
% % % 
% % %            Write to the GIF File 
% % %           if ii==1
% % %               imwrite(imind,cm,gifname2,'gif','DelayTime',0.1,'Loopcount',Inf); 
% % %           else 
% % %               imwrite(imind,cm,gifname2,'gif','DelayTime',0.1,'WriteMode','append'); 
% % %           end 
% %     
end

save('paramater_property_mat' , 'paramater_property_mat')