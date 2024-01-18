% Randomly select the number of states (5 or 10)
n = randi([5, 10]);

% Generate system matrices
A = randn(n, n); % Ensure A is stable
while any(real(eig(A)) >= 0)
    A = randn(n, n);
end
B = randn(n, 1); % Single input assumption
C = randn(1, n); % Single output assumption
D = zeros(1, 1); % D is zero in this case

% Check for Stability
eigenvalues = eig(A);
isStable = all(real(eigenvalues) < 0);

% Design a Controller for Closed-Loop Stability
desired_poles1 = -abs(randn(1, n)) - 1;
K = place(A, B, desired_poles1);

% Closed-loop System Analysis
A = A - B * K;

% Check for Observability
O = obsv(A, C);
isObservable = rank(O) == n;

% Display results
disp(['System is Stable: ', num2str(isStable)]);
disp(['System is Observable: ', num2str(isObservable)]);

% Design the Luenberger Observer
L = place(A', C', desired_poles1)';

% Define the time step for the simulation
dt = 0.8;

% Initialize variables
t = 0:dt:40; % Time vector
u = ones(length(t), 1); % Input column vector

x0 = zeros(n, 1); % Initial state
% Define bounds for initial states
x_hat0_lower_bound = -0.5;
x_hat0_upper_bound = 0.5;

% Randomly initialize initial states within bounds
x_hat0 = (x_hat0_upper_bound - x_hat0_lower_bound) * rand(n, 1) + x_hat0_lower_bound;    % Initial estimated state

% Initialize arrays for storing values
x_values = zeros(n, length(t));
x_hat_values = zeros(n, length(t));
y_values = zeros(1, length(t));
residuals = zeros(1, length(t));
w_values = zeros(n, length(t)); % Store process noise
v_values = zeros(1, length(t)); % Store measurement noise

% Noise bounds
noise_bound_w = 0.05;
noise_bound_v = 0.01;


% Define matrix exponential function
expAt = @(t, A) expm(A * t);

% Define the initial estimation error or baseline threshold level
e0 = abs(x_hat0);

% Initialize the dynamic threshold value
dynamic_thresh = zeros(1, length(t));

% Main loop for simulation
for k = 1:length(t)
    currentTime = t(k);

    % Generate bounded process noise
    w = randn(n, 1) * 0.01;
    w = max(min(w, noise_bound_w), -noise_bound_w); % Truncate if outside bounds
    w_values(:, k) = w; % Store the process noise

    % Generate bounded measurement noise
    v = randn * 0.01;
    v = max(min(v, noise_bound_v), -noise_bound_v); % Truncate if outside bounds
    v_values(k) = v; % Store the measurement noise

    % Integral calculations for x(t)
    integral_x1 = integral(@(tau) expAt(currentTime - tau, A) * B * u(floor(tau/dt) + 1, :), 0, currentTime, 'ArrayValued', true);
    integral_x2 = integral(@(tau) expAt(currentTime - tau, A) * w_values(:, floor(tau/dt) + 1), 0, currentTime, 'ArrayValued', true);
    
    % Update x
    x = expAt(currentTime, A) * x0 + integral_x1 + integral_x2;

    % Measurement with noise at current time
    y = C * x + v; 
    y_values(:, k) = y;

    % Integral calculation for x_hat(t)
    integral_x_hat1 = integral(@(tau) expAt(currentTime - tau, A) * B * u(floor(tau/dt) + 1, :), 0, currentTime, 'ArrayValued', true);
    integral_x_hat2 = integral(@(tau) expAt(currentTime - tau, A) * L * (y - C * x_hat_values(:, floor(tau/dt) + 1)), 0, currentTime, 'ArrayValued', true);

    % Update x_hat
    x_hat = expAt(currentTime, A) * x_hat0 + integral_x_hat1 + integral_x_hat2;

    % Store the states
    x_values(:, k) = x;
    x_hat_values(:, k) = x_hat;

    % Calculate and store the residual
    residuals(k) = y - C * x_hat;

    % Update the dynamic threshold using the specified formula
    integral_term = integral(@(tau) abs(C * expm((A - L * C) * tau)) * ...
        (abs(w_values(:, floor(tau/dt) + 1) - L * v_values(floor(tau/dt) + 1))), 0, currentTime, 'ArrayValued', true);

    dynamic_thresh(k) = abs(C * expm((A - L * C) * currentTime)) * e0 + integral_term + v;
end

% Plotting the residual and dynamic threshold
figure;
plot(t, residuals, 'b', t, dynamic_thresh, 'r--');
title('Residual and Threshold over Time');
xlabel('Time (seconds)');
ylabel('Value');
legend('Residual', 'Dynamic Threshold');


% Modify the system matrix A 
A_faulty = A;  % Copy the original A matrix
A_faulty(1,1) = A_faulty(1,1) * 2;  % Modification to simulate a fault

% Reset the initial state 
x_values = zeros(n, length(t)); 
x_hat_values = zeros(n, length(t)); 

% Reset residuals 
residuals_faulty = zeros(1, length(t));

for k = 1:length(t)
    currentTime = t(k);

    % Generate bounded process noise
    w = randn(n, 1) * 0.01;
    w = max(min(w, noise_bound_w), -noise_bound_w); % Truncate if outside bounds
    w_values(:, k) = w; % Store the process noise

    % Generate bounded measurement noise
    v = randn * 0.01;
    v = max(min(v, noise_bound_v), -noise_bound_v); % Truncate if outside bounds
    v_values(k) = v; % Store the measurement noise

    % Integral calculations for x(t)
    integral_x1 = integral(@(tau) expAt(currentTime - tau, A_faulty) * B * u(floor(tau/dt) + 1, :), 0, currentTime, 'ArrayValued', true);
    integral_x2 = integral(@(tau) expAt(currentTime - tau, A_faulty) * w_values(:, floor(tau/dt) + 1), 0, currentTime, 'ArrayValued', true);
    
    % Update x
    x = expAt(currentTime, A_faulty) * x0 + integral_x1 + integral_x2;

    % Measurement with noise at current time
    y = C * x + v; 
    y_values(:, k) = y;

    % Integral calculation for x_hat(t)
    integral_x_hat1 = integral(@(tau) expAt(currentTime - tau, A) * B * u(floor(tau/dt) + 1, :), 0, currentTime, 'ArrayValued', true);
    integral_x_hat2 = integral(@(tau) expAt(currentTime - tau, A) * L * (y - C * x_hat_values(:, floor(tau/dt) + 1)), 0, currentTime, 'ArrayValued', true);

    % Update x_hat
    x_hat = expAt(currentTime, A) * x_hat0 + integral_x_hat1 + integral_x_hat2;

    % Store the states
    x_values(:, k) = x;
    x_hat_values(:, k) = x_hat;

    % Calculate and store the residual
    residuals_faulty(k) = y - C * x_hat;

   % Check if the current residual exceeds the threshold and trigger an alarm
    if abs(residuals_faulty(k)) > dynamic_thresh(k)
        disp(['Alarm at time ', num2str(t(k)), '! Potential fault detected.']);
    end
end
% Plotting residuals and thresholds post-fault
figure;
plot(t, residuals_faulty, 'b', t, dynamic_thresh, 'r--');
xlabel('Time (seconds)');
ylabel('Magnitude');
title('Residuals and Threshold(Post-Fault)');
legend('Residual', 'Dynamic Threshold');
