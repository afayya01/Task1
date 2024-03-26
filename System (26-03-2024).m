n = 10;

% Generate system matrices
A = [-1.72478515948009, -0.591297178875240, -0.637284287887339,  0.508932528573876, -0.468385514094128,  0.435366001354458, -0.452951441626221, -0.457433162784182, -0.504769120550096, -0.685613130885282;
    -2.70193696571514,  0.911983868192378, -1.45596302668022,   0.446831533263970,  0.336968005162467,  0.520002434235467, -1.29869025909108,   0.806379073871306, -0.827281169313197,  0.800096303752109;
     0.270515789468659, -0.0592214944694140,-0.898578600099952,  1.73842338258012,  -1.83562973957088,  -0.248642686489297,  0.0711454738061049, 0.448052706015709,  0.589148545203839, -0.158802290894055;
     2.03423986297612,  -2.91727688906664,   0.389761612374529,  1.04375650714323,  -1.08586719895539,   0.830624708398552,  1.81158684154212,  -1.25110623117262,   1.69689382170996,  -1.06553056094042;
    -0.543907403397537,  1.16539123245225,  -0.0100413290744224,-0.878669420097964, -0.486442662384228,  0.176664192734232,  0.173876140013281,  0.733467683923031, -0.601012390460593,  0.103549823201165;
    -9.19824309221565,   3.72441696164409,  -2.66102486310557,  -4.12828075442768,   3.52905799774301,   2.65615789046851,  -5.67929828003204,   2.48232690349735,  -3.96856724795157,   1.45085974625458;
    -3.87254112293040,   2.20735591826349,  -0.733799140667861, -0.925411508468179,  1.37494067510470,   2.15090360801394,  -3.18053012458922,   0.864282780528303, -0.428474971302745,  0.574854933104201;
     1.66957794775582,   1.51703107421690,   0.305618690375098, -0.466906400831278,  0.277980118648788, -0.976712660819379,  1.34443949104452,  -1.67603835853561,   0.642402686267156, -0.396333527867103;
     0.0632230394861677, 1.85941814459250,   1.52769587000784,   0.320890710860010,  0.620607496939867, -0.470477078575474, -1.52442521484734,  -0.0920098069032417,-0.184392167827492,  0.452474659798576;
    -2.13742738837745,  -1.00539231343055,  -1.27615931750011,  -1.88447580773315,   1.06183751333048,  0.0672641344775589,-1.66193926841296,   2.19315724645419,  -2.25565351166888,   1.07831070118986];

B = [0.229832447454909;
    0.959426783629512;
    -0.233686117359606;
    -1.20114076899222;
    0.322888403906333;
    2.71655771110621;
    1.61705469067728;
    -0.236963582094816;
    -0.00648622060183900;
    1.25706895154377];

C = [1.19890849428866, 0.452783931323356, -1.26380330642036, 0.735585434335862, -0.597577720417393, -1.76272566469281, -0.791226853629756, 0.707545542847118, 1.21033936978459, 0.811920008022525];

% Check for Stability
eigenvalues = eig(A);
isStable = all(real(eigenvalues) < 0);

% Design a Controller for Closed-Loop Stability
desired_poles1 = -abs(rand(1, n)) * 0.5;
K1 = place(A, B, desired_poles1);

% Closed-loop System Analysis
A = A - B * K1;

% Check for Observability
O = obsv(A, C);
isObservable = rank(O) == n;

% Display results
disp(['System is Stable: ', num2str(isStable)]);
disp(['System is Observable: ', num2str(isObservable)]);

% Initialization
num_simulations = 5000; % Number of simulations to run
Y = zeros(k, num_simulations); % Store measurement outputs for each simulation
Residuals = zeros(k, num_simulations); % Store residuals for each simulation
k = 200; % Number of steps

for sim = 1:num_simulations
    % Reset the initial state and noise for each simulation
    x0 = randn(n, 1); % Initial state
    u = ones(1, k); % Unit step inputs for each step
    w_cov = eye(n); % Covariance matrix for process noise
    v_cov = 1; % Covariance for measurement noise
    
    w = mvnrnd(zeros(n, 1), w_cov, k)'; % Process noise for each step
    v = mvnrnd(0, v_cov, k)'; % Measurement noise for each step
    
    x = zeros(n, k+1); % Store states over time
    y = zeros(1, k); % Store measurements over time
    x(:,1) = x0; % Initial state
    
    % State Update Equation
    for i = 1:k
        sum_Bu = zeros(n, 1);
        sum_w = zeros(n, 1);
        for j = 0:i-1
            sum_Bu = sum_Bu + (A^(i-1-j))*B*u(:,j+1);
            sum_w = sum_w + (A^(i-1-j))*w(:,j+1);
        end
        x(:,i+1) = A^i*x0 + sum_Bu + sum_w;
    end
    
    % Measurement Equation
    for i = 1:k
        sum_Bu = zeros(n, 1);
        sum_w = zeros(n, 1);
        for j = 0:i-2
            sum_Bu = sum_Bu + A^(i-2-j)*B*u(:,j+1);
            sum_w = sum_w + (A^(i-2-j))*w(:,j+1);
        end
        y(:,i) = C*((A^(i-1))*x0 + sum_Bu + sum_w) + v(i);
    end

    x_hat = zeros(n, k+1); % Estimated states
    P = cell(1, k+1); % 1 x k+1 cell array of empty error covariance matrices
    x_hat(:,1) = x0; % Initial state estimate
    P{1} = eye(n) * 1000; % Initial error covariance
    Q = w_cov; % Process noise covariance
    R = v_cov; % Measurement noise covariance
    residuals = zeros(1, k); % residuals will be 1 x k
    
    
    % Kalman Filter Loop
    for i = 1:k
        % Predict
        x_hat(:,i+1) = A*x_hat(:,i) + B*u(i); % State prediction
        P{i+1} = A*P{i}*A' + Q; % Covariance prediction
        
        % Update
        K = P{i+1}*C'/(C*P{i+1}*C' + R); % Kalman Gain Calculation
        x_hat(:,i+1) = x_hat(:,i+1) + K*(y(i) - C*x_hat(:,i+1)); % State Estimate Updat
        P{i+1} = (eye(n) - K*C)*P{i+1}; % Error Covariance Updat
        
        % Residual
        r_k = y(i) - C*x_hat(:,i+1); % Residual calculation
        % Store the residual
        residuals(i) = r_k;
    end
    % Store the output and residuals
    Y(:, sim) = y; % Assuming y contains the measurement outputs for the current simulation
    Residuals(:, sim) = residuals; % Assuming residuals contain the residuals for the current simulation
end
time_step = 50; % Example time step

% Plotting distribution for y and residual for specific time step
figure;
subplot(1,2,1);
histogram(Y(time_step, :), 'Normalization', 'probability');
title('Distribution of y');
xlabel('y value');
ylabel('Distribution');

subplot(1,2,2); % subplot for r
histogram(Residuals(time_step, :), 'Normalization', 'probability');
title('Distribution of residuals');
xlabel('Residual value');
ylabel('Distribution');
 

selected_time_steps = [50, 100, 150]; 

for i = 1:length(selected_time_steps)
    time_step = selected_time_steps(i);
    
    % Extract the data for the selected time step
    y_data = Y(time_step, :);
    residuals_data = Residuals(time_step, :);
    
    % Kernel Density Estimation for y
    figure;
    [f_y, xi_y] = ksdensity(y_data);
    plot(xi_y, f_y, 'LineWidth', 2);
    title(sprintf('Estimated PDF of Output y at Time Step %d', time_step));
    xlabel('Output y');
    ylabel('Probability Density');
    % Save the plot
    saveas(gcf, sprintf('Estimated_PDF_y_TimeStep%d.png', time_step));
    
    % Kernel Density Estimation for residuals
    figure;
    [f_r, xi_r] = ksdensity(residuals_data);
    plot(xi_r, f_r, 'LineWidth', 2);
    title(sprintf('Estimated PDF of Residuals at Time Step %d', time_step));
    xlabel('Residual');
    ylabel('Probability Density');
    % Save the plot
    saveas(gcf, sprintf('Estimated_PDF_r_TimeStep%d.png', time_step));
end

% load('my_complete_workspace.mat');  % To load workspace

% time_step = 50; 
% 
% % Extract data for the selected time step
% y_data = Y(time_step, :);
% 
% % Create the histogram and normalize it to show probability density
% histogram(y_data, 'Normalization', 'pdf'); 
% hold on; 
% 
% % Perform kernel density estimation to get the PDF
% [f_y, xi_y] = ksdensity(y_data);
% 
% % Overlay the PDF plot
% plot(xi_y, f_y, 'r', 'LineWidth', 2);
% 
% % Labels and title
% title(sprintf('Overlay of Histogram and PDF for Output y at Time Step %d', time_step));
% xlabel('Output y');
% ylabel('Probability Density');
% legend('Histogram', 'PDF');
% hold off; 
