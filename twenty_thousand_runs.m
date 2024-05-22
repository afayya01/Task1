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
num_simulations = 20000; % Number of simulations to run
Y = zeros(k, num_simulations); % Store measurement outputs for each simulation
Residuals = zeros(k, num_simulations); % Store residuals for each simulation
k = 200; % Number of steps

for sim = 1:num_simulations
    % Reset the initial state and noise for each simulation
    x0 = randn(n, 1); % Initial state
    u = ones(1, k); % Unit step inputs for each step
    w_cov = eye(n); % Covariance matrix for process noise
    v_cov = 1; % Covariance for measurement noise
    
    % Process noise for each step with bounds
    w = mvnrnd(zeros(n, 1), w_cov, k)';
    w = min(max(w, -0.5), 0.5);  % Applying bounds
    
    % Measurement noise for each step with bounds
    v = mvnrnd(0, v_cov, k)';
    v = min(max(v, -0.1), 0.1);  % Applying bounds

    
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
    Y(:, sim) = y; 
    Residuals(:, sim) = residuals;
end

save('twenty_thousand_runs.mat');


time_step = 50; 

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
    % saveas(gcf, sprintf('Estimated_PDF_y_TimeStep%d.png', time_step));

    
    % Kernel Density Estimation for residuals
    figure;
    [f_r, xi_r] = ksdensity(residuals_data);
    plot(xi_r / 1e3, f_r / 1e3, 'LineWidth', 2);
    title(sprintf('Estimated PDF of Residuals at Time Step %d', time_step));
    xlabel('Residual');
    ylabel('Probability Density');
    
    % Save the plot with a modified filename to reflect the changes
    % saveas(gcf, sprintf('Scaled_Estimated_PDF_r_TimeStep%d.png', time_step));
end


% load('twenty_thousand_runs.mat');  % To load workspace

% Task1:
%desired_poles2 = -abs(randn(1, n)) * 0.5;
L = place(A', C', desired_poles1)'; % Compute observer gain L

% Luenberger Observer and Threshold Initialization
x_hat_obs = zeros(n, k+1); % Observer estimated states
x_hat_obs(:,1) = x0; % Initial state estimate for the observer
threshold_obs = zeros(1, k); % Threshold based on observer estimates
e_0 = norm(x0 - x_hat_obs(:,1)); % Initial estimation error

% 2std & 3std thresholds implementation
% Pre-compute standard deviations of noise
std_w = std(w, 0, 2); % Process noise standard deviation, across rows (dimension 2)
std_v = std(v); % Measurement noise standard deviation


% Initialize variables for additional thresholds and overpass counts
threshold_2std = zeros(1, k);
threshold_3std = zeros(1, k);
overpass_2std = zeros(1, k);
overpass_3std = zeros(1, k);

% Main Loop for Luenberger Observer and Threshold Calculation
for k = 1:k
    % Observer state estimate update using provided equation
    sum_Bu = zeros(n, 1);
    sum_Ly_minus_Cxhat = zeros(n, 1);
    for i = 0:k-1
        sum_Bu = sum_Bu + A^(k-1-i) * B * u(i+1);
        sum_Ly_minus_Cxhat = sum_Ly_minus_Cxhat + A^(k-1-i) * L * (y(i+1) - C * x_hat_obs(:, i+1));
    end
    x_hat_obs(:, k+1) = A^k * x_hat_obs(:, 1) + sum_Bu + sum_Ly_minus_Cxhat;

    % Threshold calculation
    for j = 0:k-1
        threshold_obs(k) = threshold_obs(k) + norm(C * A^j) * (norm(w(:, j+1), 2) - norm(L * v(j+1), 2));
        threshold_2std(k) = threshold_2std(k) + norm(C * A^j) * 2 * (norm(std_w) - norm(L) * std_v);
        threshold_3std(k) = threshold_3std(k) + norm(C * A^j) * 3 * (norm(std_w) - norm(L) * std_v);
    end
    threshold_obs(k) = threshold_obs(k) + abs(v(k));
    threshold_2std(k) = threshold_2std(k) + 2 * std_v;
    threshold_3std(k) = threshold_3std(k) + 3 * std_v;
    
    % Residual calculation based on observer
    residuals(k) = y(k) - C * x_hat_obs(:, k+1);

    % Count the number of overpasses
    if residuals(k) > threshold_2std(k)
        overpass_2std(k) = 1;
    end
    if abs(residuals(k)) > threshold_3std(k)
        overpass_3std(k) = 1;
    end
end

% Calculate percentages of points that lie outside the thresholds
percent_overpass_2std = 100 * sum(overpass_2std) / k;
percent_overpass_3std = 100 * sum(overpass_3std) / k;

% Plotting Residuals and Thresholds
figure;
plot(1:k, residuals, 'LineWidth', 2);
hold on;
plot(1:k, threshold_obs, 'LineWidth', 2);
xlabel('Time Step');
ylabel('Value');
title('Residuals and Thresholds from Luenberger Observer');
legend('Residuals', 'Thresholds');
grid on;
%saveas(gcf, 'Residuals_and_Thresholds.png');  % Save the figure as PNG

% Plotting percentages
figure;
bar([percent_overpass_2std, percent_overpass_3std], 'grouped');
ylabel('Percentage (%)');
title('Percentage of Points Outside Thresholds');
set(gca, 'XTickLabel', {'2-Std Threshold', '3-Std Threshold'});
%saveas(gcf, 'Percentage_Points_Outside_Thresholds.png');  % Save the figure as PNG


% Plotting Residuals and Thresholds for 2-standard deviation
figure;
plot(1:k, residuals, 'LineWidth', 2);
hold on;
plot(1:k, threshold_2std, 'r--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Value');
title('Residuals and 2-Std Threshold from Luenberger Observer');
legend('Residuals', '2-Std Threshold');
grid on;
%saveas(gcf, 'Residuals_and_2Std_Threshold.png');  % Save the figure as PNG

% Plotting Residuals and Thresholds for 3-standard deviation
figure;
plot(1:k, residuals, 'LineWidth', 2);
hold on;
plot(1:k, threshold_3std, 'g--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Value');
title('Residuals and 3-Std Threshold from Luenberger Observer');
legend('Residuals', '3-Std Threshold');
grid on;
%saveas(gcf, 'Residuals_and_3Std_Threshold.png');  % Save the figure as PNG



% Task2:
% % Step 1: Fit Gaussian and Extract Mean & Std for All Times

% Initialize matrices to store stds and means for each time step and simulation
stds_res = zeros(k, 1);
means_res = zeros(k, 1);

% Calculate std and mean for each time step across all simulations
for t = 1:k
    % Extract all residuals up to time t across all simulations
    current_residuals = Residuals(1:t, :);

    % Calculate std and mean for the current time step across all simulations
    stds_res(t) = std(current_residuals(:));
    means_res(t) = mean(current_residuals(:));
end

% Plot the average means and stds
figure;
subplot(2,1,1);
plot(1:k, means_res, 'LineWidth', 2);
title('Average Means of Residuals');
xlabel('Time Step');
ylabel('Mean Value');

subplot(2,1,2);
plot(1:k, stds_res, 'LineWidth', 2);
title('Average Standard Deviations of Residuals');
xlabel('Time Step');
ylabel('Standard Deviation');

% Display figures
sgtitle('Analysis of Residuals Over Time');



% Step 3: Using line or plot for Compatibility
selected_time_steps = [50, 100, 150];

for i = 1:length(selected_time_steps)
    time_step = selected_time_steps(i);
    data = Residuals(time_step, :); % Assuming Residuals data for Gaussian fitting
    
    % Calculate mean and std for Gaussian PDF plotting
    mu = mean(data);
    sigma = std(data);
    
    % Gaussian PDF plotting with x and y axes scaled by 1e3
    figure;
    x_values = linspace(mu - 3*sigma, mu + 3*sigma, 1000); % Defining x axis based on mu and sigma
    pdf_values = normpdf(x_values, mu, sigma); % Gaussian PDF values
    scaled_pdf_values = pdf_values / 1e3; % Scaling PDF height by 1e3
    plot(x_values / 1e3, scaled_pdf_values, 'LineWidth', 2); % Also scaling x axis by 1e3
    hold on;
    
    ylim = [0, max(scaled_pdf_values) * 1.1]; % Adjust y-axis limits for visibility
    set(gca, 'YLim', ylim);
    
    % Plot vertical line for mean, scaling x value by 1e3
    plot([mu / 1e3, mu / 1e3], ylim, 'k--', 'LineWidth', 1.5);
    
    % Plot vertical lines for mean+2*std & mean-2*std, scaling x values
    plot([(mu + 2*sigma) / 1e3, (mu + 2*sigma) / 1e3], ylim, 'g:', 'LineWidth', 1.5);
    plot([(mu - 2*sigma) / 1e3, (mu - 2*sigma) / 1e3], ylim, 'g:', 'LineWidth', 1.5);
    
    % Plot vertical lines for mean+3*std & mean-3*std, also scaling by 1e3
    plot([(mu + 3*sigma) / 1e3, (mu + 3*sigma) / 1e3], ylim, 'm-.', 'LineWidth', 1.5);
    plot([(mu - 3*sigma) / 1e3, (mu - 3*sigma) / 1e3], ylim, 'm-.', 'LineWidth', 1.5);
    
    title(['Gaussian PDF for Time Step ' num2str(time_step)]);
    xlabel('Residual');
    ylabel('Probability Density');
    legend('Gaussian PDF', 'Mean', 'Mean + 2*Std', 'Mean - 2*Std', 'Mean + 3*Std', 'Mean - 3*Std', 'Location', 'Best');
    
    hold off;
    % Save the plot
    %saveas(gcf, sprintf('Scaled_Gaussian_PDF_Residuals_TimeStep%d.png', time_step));
end